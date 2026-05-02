"""Registry of models: (model name, dataset name, pipeline class)."""

from typing import Protocol, TypeAlias

import pandas as pd

from pipelines.ball_impact import Pipeline as BallImpactPipeline

__all__ = ["TrainingPipeline", "MODELS"]


class TrainingPipeline(Protocol):
    """Each pipeline module defines a ``Pipeline`` class implementing this protocol."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw training data into feature columns."""
        ...

    def train(self, features: pd.DataFrame) -> object:
        """Fit and return a scikit-learn–compatible or XGBoost estimator."""
        ...


RegistryEntry: TypeAlias = tuple[str, str, type[TrainingPipeline]]

MODELS: list[RegistryEntry] = [
    ("ball_impact_v1", "ball_impact_training", BallImpactPipeline),
]
