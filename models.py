"""Shared models used across metric pipelines."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Outcome of a single pipeline run."""

    pipeline_name: str
    success: bool
    message: str = ""
