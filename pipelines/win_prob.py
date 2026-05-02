"""Ball-by-ball **team** win probability (limited overs).

Features are match state only (runs, wickets, balls, chase math). There are no
player IDs here — if you rank batters/bowlers using predicted win-probability
changes at each ball, results mix **player skill** with **team strength** and
match situation; expect systematic bias unless you residualize or model players.

Training rows must align 1:1 with ``df.index``: feature columns are built from
``pd.DataFrame(index=df.index)`` so row ``i`` matches delivery ``i`` in ``df``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from utils.features import *


def _calibration_cv(y: np.ndarray, *, max_folds: int = 5) -> int:
    """Pick a StratifiedKFold fold count that avoids empty classes per fold."""
    y_int = y.astype(int)
    n = len(y_int)
    pos = int((y_int == 1).sum())
    neg = n - pos
    cap = max(pos, neg)
    cv = min(max_folds, cap, n // 2) if n >= 4 else 2
    return max(2, cv)


def _calibrated_logistic_pipeline(y: np.ndarray) -> CalibratedClassifierCV:
    """StandardScaler + logistic regression; probabilities calibrated (sigmoid CV)."""
    base = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(random_state=42, max_iter=2000),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=_calibration_cv(y),
        n_jobs=-1,
    )


def _print_training_metrics(name: str, model, x: np.ndarray, y: np.ndarray) -> None:
    """Print simple in-sample metrics for quick pipeline feedback."""
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1]
    acc = accuracy_score(y, y_pred)
    ll = log_loss(y, y_proba)
    try:
        auc = roc_auc_score(y, y_proba)
        auc_msg = f"{auc:.4f}"
    except ValueError:
        auc_msg = "n/a (single class)"
    print(
        f"[{name}] rows={len(y)} accuracy={acc:.4f} log_loss={ll:.4f} roc_auc={auc_msg}",
        flush=True,
    )


def _sanitize_training_rows(features_df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    """Drop rows with NaN/inf in features or label to keep sklearn fit stable."""
    cleaned = features_df.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("No valid rows after filtering NaN/inf values from computed features.")
    if cleaned[label_col].nunique() < 2:
        print(
            f"[warn] {label_col} has a single class after cleaning; model may be degenerate.",
            flush=True,
        )
    return cleaned


class FirstInningsWinProbPipeline:
    """Training pipeline with ``compute_features`` + ``train``."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Must share df.index so each row is the same delivery as in df (critical).
        features_df = pd.DataFrame(index=df.index)
        features_df["innings_runs"] = innings_runs(df, cumulative=True)
        features_df["innings_wickets"] = innings_wickets(df, cumulative=True)
        features_df["innings_legal_balls"] = innings_legal_balls(df)
        denom = features_df["innings_legal_balls"].replace(0, np.nan)
        features_df["runs_per_ball"] = features_df["innings_runs"] / denom
        features_df["batting_team_won"] = batting_team_won(df)
        # Explicit innings 1 only (not ``with_target == 0``, which would include innings 3+ in Tests).
        features_df = features_df.loc[df["innings"] == 1]
        return _sanitize_training_rows(features_df, label_col="batting_team_won")

    def train(self, features: pd.DataFrame) -> CalibratedClassifierCV:
        x = features.drop(columns=["batting_team_won"]).to_numpy(dtype=float)
        y = features["batting_team_won"].to_numpy(dtype=float)
        model = _calibrated_logistic_pipeline(y)
        model.fit(x, y)
        _print_training_metrics("first_innings_win_prob", model, x, y)
        return model


class SecondInningsWinProbPipeline:
    """Training pipeline with ``compute_features`` + ``train``."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = pd.DataFrame(index=df.index)
        legal_before = innings_legal_balls(df)
        target = with_target(df)
        features_df["runs_required"] = target - innings_runs(df, cumulative=True)
        features_df["innings_wickets"] = innings_wickets(df, cumulative=True)
        features_df["innings_legal_balls"] = legal_before
        total_legal_in_innings = legal_before.groupby(
            [df["match_id"], df["innings"]], sort=False
        ).transform("max") + 1
        balls_remaining = total_legal_in_innings - legal_before
        features_df["runs_required_per_ball"] = features_df["runs_required"] / balls_remaining.replace(
            0, np.nan
        )
        features_df["batting_team_won"] = batting_team_won(df)

        features_df = features_df.loc[df["innings"] == 2]
        features_df = features_df[features_df["runs_required"] > 0]
        return _sanitize_training_rows(features_df, label_col="batting_team_won")

    def train(self, features: pd.DataFrame) -> CalibratedClassifierCV:
        x = features.drop(columns=["batting_team_won"]).to_numpy(dtype=float)
        y = features["batting_team_won"].to_numpy(dtype=float)
        model = _calibrated_logistic_pipeline(y)
        model.fit(x, y)
        _print_training_metrics("second_innings_win_prob", model, x, y)
        return model
