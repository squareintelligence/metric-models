"""Registry of models: (model name, dataset name, pipeline class)."""

from typing import Protocol, TypeAlias

import pandas as pd

from pipelines.win_prob import FirstInningsWinProbPipeline, SecondInningsWinProbPipeline

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
    ("ipl_from_2023/first_innings/win_prob_v8", "ipl_from_2023", FirstInningsWinProbPipeline),
    ("ipl_from_2023/second_innings/win_prob_v8", "ipl_from_2023", SecondInningsWinProbPipeline),
]
