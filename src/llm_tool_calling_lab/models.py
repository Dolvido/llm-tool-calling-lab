"""Validated specialist execution, persistent evidence, and private evaluation."""

import json
import multiprocessing as mp
from pathlib import Path
import re
import time
import uuid
import warnings

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .catalog import build_estimator, catalog_entries, describe_methods
from .contracts import RunResult, TaskSpec
from .data import DatasetStore, json_write, partition_fingerprint, split_indices, validate_task

RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")


def _predictions(indices, values):
    return [{"row_id": str(int(row)), "prediction": value.item() if hasattr(value, "item") else value}
            for row, value in zip(indices, values)]


def _validated_vector(values, expected_rows: int, label: str, categories: int | None = None):
    """Every adapter must return one finite numeric value for each input row."""
    vector = np.asarray(values)
    if vector.ndim != 1 or len(vector) != expected_rows:
        raise ValueError(f"{label} must be a one-dimensional vector with exactly {expected_rows} values.")
    if not np.issubdtype(vector.dtype, np.number) or np.iscomplexobj(vector) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain only finite real numbers.")
    if categories is not None:
        if not np.equal(vector, np.floor(vector)).all() or np.any(vector < 0) or np.any(vector >= categories):
            raise ValueError(f"{label} must contain integer category codes from zero through {categories - 1}.")
        return vector.astype(int)
    return vector.astype(float)


def _execute(root: str, catalog: str, task_payload: dict, method_id: str, seed: int, run_id: str) -> dict:
    started = time.monotonic()
    task = TaskSpec.model_validate(task_payload)
    run_directory = Path(root) / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    store = DatasetStore(Path(root))
    matrix, target, scoring = validate_task(store, task)
    if task.family == "anomaly":
        training = np.arange(len(matrix))
        validation = testing = np.array([], dtype=int)
    else:
        training, validation, testing = split_indices(len(matrix), seed, target if task.family == "classification" else None)
    target_encoder = None
    if task.family == "classification":
        target_encoder = LabelEncoder().fit(target[training])
        target = target_encoder.transform(target)
    if np.isnan(matrix[training]).all(axis=0).any():
        raise ValueError("A feature is entirely missing in training rows; revise the input.")
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    fit_features = preprocessor.fit_transform(matrix[training])
    estimator = build_estimator(method_id, catalog, seed, task.n_clusters)
    observed_warnings = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            if target is None:
                estimator.fit(fit_features)
            else:
                estimator.fit(fit_features, target[training])
        observed_warnings.extend(str(item.message) for item in captured)
    metrics, summary, private = {}, {}, {}
    summary.update({"family": task.family, "seed": seed, "features": task.features, "target": task.target,
                    "parameters": catalog_entries(catalog)[method_id]["parameters"],
                    "partition_counts": {"training": len(training), "validation": len(validation), "test": len(testing)},
                    "partition_fingerprints": {name: partition_fingerprint(indices) for name, indices in (("training", training), ("validation", validation), ("test", testing))}})
    private.update({"training_indices": training.tolist(), "validation_indices": validation.tolist(), "test_indices": testing.tolist(),
                    "imputer_statistics": preprocessor.named_steps["imputer"].statistics_.tolist(), "scaler_mean": preprocessor.named_steps["scaler"].mean_.tolist()})
    with threadpool_limits(limits=1):
        if task.family == "anomaly":
            score_values = -_validated_vector(estimator.score_samples(preprocessor.transform(scoring)), len(scoring), "Anomaly scores")
            ranking = np.argsort(-score_values, kind="stable")
            public_predictions = [{"row_id": str(int(row)), "score": float(score_values[row])} for row in ranking]
            summary.update({"ranking": public_predictions[:10], "score_direction": "Larger scores mean more unusual relative to the reference data.",
                            "score_distribution": {"minimum": float(score_values.min()), "median": float(np.median(score_values)), "maximum": float(score_values.max())},
                            "scoring_rows": len(scoring), "limitation": "No labels were used. Ranking is not an accuracy estimate or proof of normality."})
            private["ranking"] = [str(int(row)) for row in ranking]
            private["scores"] = score_values.tolist()
        else:
            validation_features = preprocessor.transform(matrix[validation])
            test_features = preprocessor.transform(matrix[testing])
            category_count = 2 if task.family == "classification" else task.n_clusters if task.family == "clustering" else None
            validation_predictions = _validated_vector(estimator.predict(validation_features), len(validation), "Validation predictions", category_count)
            test_predictions = _validated_vector(estimator.predict(test_features), len(testing), "Test predictions", category_count)
            public_values = target_encoder.inverse_transform(validation_predictions) if target_encoder is not None else validation_predictions
            public_predictions = _predictions(validation, public_values)
            private["test_predictions"] = test_predictions.tolist()
            summary["validation_predictions"] = public_predictions[:10]
            if task.family == "regression":
                metrics["validation_mae"] = float(mean_absolute_error(target[validation], validation_predictions))
                constant = float(np.median(target[training]))
                summary["baseline_strategy"] = "training_target_median"
                metrics["baseline_validation_mae"] = float(mean_absolute_error(target[validation], np.full(len(validation), constant)))
                private["test_metrics"] = {"test_mae": float(mean_absolute_error(target[testing], test_predictions)), "baseline_test_mae": float(mean_absolute_error(target[testing], np.full(len(testing), constant)))}
            elif task.family == "classification":
                metrics["validation_balanced_accuracy"] = float(balanced_accuracy_score(target[validation], validation_predictions))
                labels = estimator.classes_.tolist()
                category_names = target_encoder.inverse_transform(estimator.classes_).tolist()
                summary["classes"] = category_names
                summary["validation_confusion_matrix"] = confusion_matrix(target[validation], validation_predictions, labels=labels).tolist()
                summary["validation_class_counts"] = {name: int(np.sum(target[validation] == label)) for name, label in zip(category_names, labels)}
                private["test_metrics"] = {"test_balanced_accuracy": float(balanced_accuracy_score(target[testing], test_predictions)), "test_confusion_matrix": confusion_matrix(target[testing], test_predictions, labels=labels).tolist()}
            else:
                groups = np.unique(validation_predictions)
                defined = 1 < len(groups) < len(validation_predictions)
                metrics["validation_silhouette"] = float(silhouette_score(validation_features, validation_predictions, sample_size=min(2000, len(validation)), random_state=seed)) if defined else None
                summary["diagnostic_eligible"] = defined
                summary["n_clusters"] = task.n_clusters
                summary["validation_group_counts"] = {str(int(group)): int(np.sum(validation_predictions == group)) for group in groups}
                summary["limitation"] = "Silhouette measures geometric separation; groups have no established real-world meaning."
                if not defined:
                    observed_warnings.append("Validation silhouette is undefined; candidate is ineligible for diagnostic ranking.")
    if len(set(private.get("ranking", []))) != len(private.get("ranking", [])):
        raise ValueError("Invalid duplicate ranking IDs.")
    json_write(run_directory / "predictions.json", public_predictions)
    json_write(run_directory / "private.json", private)
    joblib.dump({"preprocessor": preprocessor, "estimator": estimator, "target_encoder": target_encoder, "task": task.model_dump()}, run_directory / "model.joblib")
    result = RunResult(run_id=run_id, method_id=method_id, task=task, status="ok", metrics=metrics, summary=summary,
                       warnings=observed_warnings, artifact_paths={"result": str(run_directory / "result.json"), "predictions": str(run_directory / "predictions.json")}, elapsed_seconds=time.monotonic() - started)
    payload = result.model_dump(mode="json")
    json_write(run_directory / "result.json", payload)
    return payload


def _worker(connection, root, catalog, task_payload, method_id, seed, run_id):
    try:
        connection.send({"result": _execute(root, catalog, task_payload, method_id, seed, run_id)})
    except Exception as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


class ModelService:
    def __init__(self, root: Path, catalog: str = "default"):
        self.root = Path(root).resolve()
        self.catalog = catalog
        catalog_entries(catalog)
        self.store = DatasetStore(self.root)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    def register_csv(self, path: Path) -> str:
        return self.store.register_csv(path)

    def inspect_dataset(self, dataset_id: str) -> dict:
        return self.store.inspect(dataset_id)

    def list_methods(self, family: str | None = None) -> list[dict]:
        return describe_methods(self.catalog, family)

    def run_candidate(self, task: TaskSpec, method_id: str, seed: int = 0, timeout_s: float = 60) -> RunResult:
        task = TaskSpec.model_validate(task)
        started = time.monotonic()
        run_id = "run_" + uuid.uuid4().hex
        run_directory = self.root / "runs" / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        process = None
        parent = child = None
        try:
            entries = catalog_entries(self.catalog)
            if method_id not in entries or entries[method_id]["family"] != task.family:
                raise ValueError("Unknown method or method does not support this task family.")
            if not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("Seed must be a nonnegative 32-bit integer.")
            if not np.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > 180:
                raise ValueError("Timeout must be positive and at most 180 seconds.")
            # Validate before process launch so invalid inputs fail immediately.
            validate_task(self.store, task)
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Model execution exceeded its time limit.")
            context = mp.get_context("spawn")
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_worker, args=(child, str(self.root), self.catalog, task.model_dump(mode="json"), method_id, seed, run_id), daemon=True)
            process.start()
            child.close()
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0 or not parent.poll(remaining):
                raise TimeoutError("Model execution exceeded its time limit.")
            message = parent.recv()
            process.join(timeout=1)
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("Model execution exceeded its time limit.")
            if "error" in message:
                raise RuntimeError(message["error"])
            result = RunResult.model_validate(message["result"])
            result.elapsed_seconds = time.monotonic() - started
            json_write(run_directory / "result.json", result.model_dump(mode="json"))
            return result
        except Exception as exc:
            # Stop writers before saving failure, so a late worker cannot replace
            # a timeout artifact with a successful result after the budget ended.
            if process is not None and process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            result = RunResult(run_id=run_id, method_id=method_id, task=task, status="error", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started,
                               artifact_paths={"result": str(run_directory / "result.json")})
            json_write(run_directory / "result.json", result.model_dump(mode="json"))
            return result
        finally:
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                process.close()
            if parent is not None:
                parent.close()
            if child is not None:
                child.close()

    def _run_directory(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError("Invalid run reference.")
        result = self.root / "runs" / run_id
        if not (result / "result.json").is_file():
            raise ValueError("Unknown run reference.")
        return result

    def retrieve_result(self, run_id: str) -> dict:
        # Deliberate allowlist: no private evaluator record or fitted object is exposed.
        payload = json.loads((self._run_directory(run_id) / "result.json").read_text(encoding="utf-8"))
        return RunResult.model_validate(payload).model_dump(mode="json")

    def compare_results(self, run_ids: list[str]) -> dict:
        if not 2 <= len(run_ids) <= 8 or len(set(run_ids)) != len(run_ids):
            raise ValueError("Compare between two and eight distinct run references.")
        results = [self.retrieve_result(run_id) for run_id in run_ids]
        def signature(result):
            task = result["task"]
            return tuple(json.dumps(task.get(key), sort_keys=True) for key in ("family", "dataset_id", "features", "target", "scoring_dataset_id", "n_clusters")) + (json.dumps(result["summary"].get("partition_fingerprints"), sort_keys=True),)
        if any(result["status"] != "ok" for result in results):
            raise ValueError("Only successful runs can be compared.")
        if len({signature(result) for result in results}) != 1:
            raise ValueError("Results must share task, data roles, features, and partitions.")
        family = results[0]["task"]["family"]
        comparison = {"family": family, "run_ids": run_ids, "results": [{"run_id": result["run_id"], "method_id": result["method_id"], "metrics": result["metrics"]} for result in results]}
        if family == "anomaly":
            top = [{row["row_id"] for row in result["summary"]["ranking"][:4]} for result in results]
            comparison["top_four_overlap"] = len(set.intersection(*top))
            comparison["top_four_row_ids"] = {result["run_id"]: [row["row_id"] for row in result["summary"]["ranking"][:4]] for result in results}
            comparison["limitation"] = "Ranking overlap measures agreement, not accuracy. No winning method is established without labels."
        else:
            metric = {"regression": "validation_mae", "classification": "validation_balanced_accuracy", "clustering": "validation_silhouette"}[family]
            eligible = [result for result in results if result["metrics"].get(metric) is not None]
            comparison["metric"] = metric
            comparison["preferred_run_id"] = (min(eligible, key=lambda result: result["metrics"][metric]) if family == "regression" else max(eligible, key=lambda result: result["metrics"][metric]))["run_id"] if eligible else None
            comparison["limitation"] = "Preference uses this validation diagnostic only; it does not establish broader superiority."
        return comparison

    def evaluation_metrics(self, run_id: str, truth: dict | None = None) -> dict:
        result = self.retrieve_result(run_id)
        if result["status"] != "ok":
            return {"status": "error", "error": result["error"]}
        private = json.loads((self._run_directory(run_id) / "private.json").read_text(encoding="utf-8"))
        family = result["task"]["family"]
        if family in {"regression", "classification"}:
            return private["test_metrics"]
        truth = truth or {}
        if family == "anomaly":
            if "anomaly_ids" not in truth and "labels" not in truth:
                return {}
            labels = truth.get("labels", {})
            if isinstance(labels, list):
                labels = dict(enumerate(labels))
            anomaly_ids = {str(row) for row in truth.get("anomaly_ids", [row for row, label in labels.items() if label])}
            top = private["ranking"][:4]
            score = len(set(top) & anomaly_ids) / 4 if len(top) == 4 and len(set(top)) == 4 else 0.0
            return {"precision_at_4": score}
        labels = truth.get("labels")
        if labels is None:
            return {}
        if isinstance(labels, list):
            labels = {str(index): label for index, label in enumerate(labels)}
        else:
            labels = {str(index): label for index, label in labels.items()}
        try:
            selected = [labels[str(index)] for index in private["test_indices"]]
        except KeyError as exc:
            raise ValueError("Evaluator labels must cover all test row IDs.") from exc
        return {"test_adjusted_rand_index": float(adjusted_rand_score(selected, private["test_predictions"]))}
