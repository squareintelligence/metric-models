"""Example pipeline: replace feature logic and estimator with your metric."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from utils.features import *


def _print_training_metrics(name: str, model: SklearnPipeline, x: pd.DataFrame, y: pd.Series) -> None:
    """Print simple in-sample metrics for quick pipeline feedback."""
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1]
    acc = accuracy_score(y, y_pred)
    ll = log_loss(y, y_proba)
    try:
        auc = roc_auc_score(y, y_proba)
        auc_msg = f"{auc:.4f}"
    except ValueError:
        # Happens when y has a single class in the training slice.
        auc_msg = "n/a (single class)"
    print(
        f"[{name}] rows={len(y)} accuracy={acc:.4f} log_loss={ll:.4f} roc_auc={auc_msg}",
        flush=True,
    )

class FirstInningsWinProbPipeline:
    """Training pipeline with ``compute_features`` + ``train``."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = pd.DataFrame()
        features_df["innings_runs"] = innings_runs(df, cumulative=True)
        features_df["innings_wickets"] = innings_wickets(df, cumulative=True)
        features_df["innings_legal_balls"] = innings_legal_balls(df)
        features_df["runs_per_ball"] = features_df["innings_runs"] / features_df["innings_legal_balls"]
        features_df["batting_team_won"] = batting_team_won(df)
        first_innings_mask = with_target(df) == 0

        features_df = features_df[first_innings_mask]

        return features_df

    def train(self, features: pd.DataFrame) -> SklearnPipeline:
        x = features.drop(columns=["batting_team_won"]).to_numpy(dtype=float)
        y = features["batting_team_won"].to_numpy(dtype=float)
        model = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(random_state=0)),
            ]
        )
        model.fit(x, y)
        _print_training_metrics("first_innings_win_prob", model, x, y)
        return model


class SecondInningsWinProbPipeline:
    """Training pipeline with ``compute_features`` + ``train``."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = pd.DataFrame()
        innings_balls = innings_legal_balls(df)
        max_innings_balls = innings_balls.max()
        features_df["target"] = with_target(df)
        features_df["runs_required"] = features_df["target"] -innings_runs(df, cumulative=True)
        features_df["innings_wickets"] = innings_wickets(df, cumulative=True)
        features_df["innings_legal_balls"] = innings_legal_balls(df)
        features_df["runs_required_per_ball"] = features_df["runs_required"] / (max_innings_balls - features_df["innings_legal_balls"])
        features_df["batting_team_won"] = batting_team_won(df)

        features_df = features_df[features_df["runs_required"] > 0]

        return features_df

    def train(self, features: pd.DataFrame) -> SklearnPipeline:
        x = features.drop(columns=["batting_team_won"]).to_numpy(dtype=float)
        y = features["batting_team_won"].to_numpy(dtype=float)
        model = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(random_state=0)),
            ]
        )
        model.fit(x, y)
        _print_training_metrics("second_innings_win_prob", model, x, y)
        return model
