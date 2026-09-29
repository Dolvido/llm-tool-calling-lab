"""Local CSV registration and strictly bounded numerical task validation."""

import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_BYTES = 10 * 1024 * 1024
DATASET_ID = re.compile(r"ds_[0-9a-f]{20}\Z")


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class DatasetStore:
    def __init__(self, root: Path):
        self.root = Path(root) / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)

    def register_csv(self, path: Path) -> str:
        path = Path(path)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("CSV exceeds the 10 MB size limit.")
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError("CSV exceeds the 10 MB size limit.")
        self._read(raw)
        dataset_id = "ds_" + hashlib.sha256(raw).hexdigest()[:20]
        destination = self.root / f"{dataset_id}.csv"
        if not destination.exists():
            destination.write_bytes(raw)
        return dataset_id

    @staticmethod
    def _read(raw: bytes) -> pd.DataFrame:
        try:
            decoded = raw.decode("utf-8-sig")
            reader = csv.reader(io.StringIO(decoded), strict=True)
            header = next(reader)
            if not header or any(not column.strip() for column in header) or len(set(header)) != len(header):
                raise ValueError("CSV requires nonempty, unique column names.")
            if len(header) > 256 or any(len(column) > 128 for column in header):
                raise ValueError("CSV schema is too wide or column names are too long.")
            if any(row and len(row) != len(header) for row in reader):
                raise ValueError("Every CSV row must match the header column count.")
            frame = pd.read_csv(io.StringIO(decoded), on_bad_lines="error")
        except (UnicodeError, StopIteration, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError("Provide a nonempty, well-formed UTF-8 CSV.") from exc
        if len(frame) < 1 or len(frame) > 10_000:
            raise ValueError("CSV must contain between 1 and 10,000 rows.")
        return frame

    def load(self, dataset_id: str) -> pd.DataFrame:
        if not isinstance(dataset_id, str) or not DATASET_ID.fullmatch(dataset_id):
            raise ValueError("Invalid dataset reference.")
        path = self.root / f"{dataset_id}.csv"
        if not path.is_file():
            raise ValueError("Unknown dataset reference.")
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES or "ds_" + hashlib.sha256(raw).hexdigest()[:20] != dataset_id:
            raise ValueError("Registered dataset changed after registration.")
        return self._read(raw)

    def inspect(self, dataset_id: str) -> dict:
        frame = self.load(dataset_id)
        return {"dataset_id": dataset_id, "rows": len(frame), "columns": [
            {"name": str(column), "dtype": str(frame[column].dtype), "numeric": bool(pd.api.types.is_numeric_dtype(frame[column])), "missing": int(frame[column].isna().sum())}
            for column in frame.columns], "row_id_format": "Zero-based row positions represented as strings.",
            "note": "Schema only: no row values or target distributions are disclosed."}


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not 2 <= len(features) <= 30 or len(set(features)) != len(features):
        raise ValueError("Select 2–30 distinct numerical feature columns.")
    if any(column not in frame.columns for column in features):
        raise ValueError("One or more selected feature columns do not exist.")
    selected = frame[features]
    if any(not pd.api.types.is_numeric_dtype(selected[column]) for column in features):
        raise ValueError("Only numerical features are supported; encode or remove nonnumeric columns.")
    matrix = selected.to_numpy(dtype=float)
    if np.isinf(matrix).any():
        raise ValueError("Infinite feature values are not supported.")
    if np.isnan(matrix).all(axis=0).any():
        raise ValueError("An entirely missing feature cannot be imputed.")
    return matrix


def validate_task(store: DatasetStore, task):
    frame = store.load(task.dataset_id)
    if not 100 <= len(frame) <= 10_000:
        raise ValueError("Fitting/reference data requires 100–10,000 rows.")
    if task.target is not None and task.target in task.features:
        raise ValueError("The target column cannot also be a feature.")
    matrix = feature_matrix(frame, task.features)
    target = None
    if task.family in {"regression", "classification"}:
        if not task.target or task.target not in frame.columns:
            raise ValueError("A supervised task requires an existing target column.")
        series = frame[task.target]
        if series.isna().any():
            raise ValueError("Missing target values are not supported.")
        if pd.api.types.is_numeric_dtype(series) and not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError("Infinite target values are not supported.")
        if task.family == "regression":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("Regression requires a numerical target.")
            target = series.to_numpy(dtype=float)
        else:
            counts = series.value_counts()
            if len(counts) != 2 or counts.min() < 5:
                raise ValueError("Classification requires exactly two categories with at least five rows each.")
            # Preserve category identity while accepting numeric labels such as 0.0/1.0.
            target = series.astype(str).to_numpy()
    elif task.target is not None:
        raise ValueError("Unsupervised tasks must omit the target column.")
    scoring = None
    if task.family == "anomaly":
        if not task.scoring_dataset_id or task.scoring_dataset_id == task.dataset_id:
            raise ValueError("Anomaly detection requires a separate, fresh scoring dataset.")
        scoring = feature_matrix(store.load(task.scoring_dataset_id), task.features)
    elif task.scoring_dataset_id is not None:
        raise ValueError("A separate scoring dataset is supported only for anomaly detection in v0.1.")
    if task.family == "clustering" and not 2 <= task.n_clusters <= 10:
        raise ValueError("Request between 2 and 10 exploratory groups.")
    return matrix, target, scoring


def split_indices(size: int, seed: int, target=None):
    indices = np.arange(size)
    training, remainder = train_test_split(indices, test_size=0.4, random_state=seed, stratify=target)
    remainder_target = target[remainder] if target is not None else None
    validation, testing = train_test_split(remainder, test_size=0.5, random_state=seed, stratify=remainder_target)
    return training, validation, testing


def partition_fingerprint(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
