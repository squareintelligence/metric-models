"""S3 and storage settings from the environment (CI-friendly)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """Paths are S3 key prefixes (no leading ``s3://``; bucket is separate)."""

    bucket: str
    aws_region: str
    datasets_prefix: str
    models_prefix: str

    @classmethod
    def from_environ(cls) -> "StorageSettings":
        bucket = os.environ.get("S3_BUCKET", "").strip()
        if not bucket:
            raise RuntimeError("S3_BUCKET must be set (S3 bucket for datasets and models).")
        region = (
            os.environ.get("AWS_DEFAULT_REGION", "").strip()
            or os.environ.get("AWS_REGION", "").strip()
        )
        if not region:
            raise RuntimeError("Set AWS_DEFAULT_REGION or AWS_REGION for boto3.")
        datasets_prefix = os.environ.get("S3_DATASETS_PREFIX", "datasets/").strip()
        models_prefix = os.environ.get("S3_MODELS_PREFIX", "models/").strip()
        return cls(
            bucket=bucket,
            aws_region=region,
            datasets_prefix=datasets_prefix,
            models_prefix=models_prefix,
        )
