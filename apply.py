"""Train and upload missing models to S3 using ``models.MODELS`` registry."""

from __future__ import annotations

import io
import sys
from typing import Any

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError

from config import StorageSettings
from models import MODELS


def _normalize_prefix(prefix: str) -> str:
    p = prefix.strip().strip("/")
    return f"{p}/" if p else ""


def _dataset_key(settings: StorageSettings, dataset_name: str) -> str:
    base = _normalize_prefix(settings.datasets_prefix)
    return f"{base}{dataset_name}.csv"


def _model_key(settings: StorageSettings, model_name: str) -> str:
    base = _normalize_prefix(settings.models_prefix)
    return f"{base}{model_name}.joblib"


def _object_exists(s3: Any, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _load_dataset_csv(s3: Any, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    buf = io.BytesIO(body)
    return pd.read_csv(buf)


def _upload_model(s3: Any, bucket: str, key: str, model: object) -> None:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def main() -> None:
    settings = StorageSettings.from_environ()
    print(f"Bucket: {settings.bucket}", flush=True)
    print(f"Region: {settings.aws_region}", flush=True)
    print(f"Datasets prefix: {settings.datasets_prefix!r}", flush=True)
    print(f"Models prefix: {settings.models_prefix!r}", flush=True)

    session = boto3.Session(region_name=settings.aws_region)
    s3 = session.client("s3")

    if not MODELS:
        print("No entries in models.MODELS; nothing to do.", flush=True)
        return

    for model_name, dataset_name, pipeline_cls in MODELS:
        mkey = _model_key(settings, model_name)
        print(f"\n=== Model {model_name!r} (dataset {dataset_name!r}) ===", flush=True)
        if _object_exists(s3, settings.bucket, mkey):
            print(f"  Skip: already present at s3://{settings.bucket}/{mkey}", flush=True)
            continue

        dkey = _dataset_key(settings, dataset_name)
        print(f"  Loading dataset s3://{settings.bucket}/{dkey}", flush=True)
        try:
            raw = _load_dataset_csv(s3, settings.bucket, dkey)
        except ClientError as e:
            print(f"  ERROR: failed to load dataset: {e}", flush=True)
            raise

        print(f"  Rows: {len(raw)}, columns: {list(raw.columns)}", flush=True)
        pipeline = pipeline_cls()
        print("  compute_features…", flush=True)
        features = pipeline.compute_features(raw)
        print(f"  Feature matrix shape: {features.shape}", flush=True)
        print("  train…", flush=True)
        estimator = pipeline.train(features)
        print(f"  Uploading model to s3://{settings.bucket}/{mkey}", flush=True)
        _upload_model(s3, settings.bucket, mkey, estimator)
        print("  Done.", flush=True)

    print("\nAll pending models processed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("apply.py failed.", file=sys.stderr, flush=True)
        raise
