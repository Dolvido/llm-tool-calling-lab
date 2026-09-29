import json
import multiprocessing
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.metrics import silhouette_score

from llm_tool_calling_lab.contracts import TaskSpec
from llm_tool_calling_lab.models import ModelService


def make_task(tmp_path, family, catalog="default"):
    service = ModelService(tmp_path / "state", catalog=catalog)
    if family == "regression":
        matrix, target = make_regression(n_samples=150, n_features=4, noise=2, random_state=41)
    elif family == "classification":
        matrix, target = make_classification(n_samples=150, n_features=4, n_informative=3, n_redundant=0, random_state=41)
    else:
        matrix, target = make_blobs(n_samples=150, n_features=4, centers=3, random_state=41)
    features = [f"x{index}" for index in range(4)]
    frame = pd.DataFrame(matrix, columns=features)
    target_column = None
    if family in {"regression", "classification"}:
        frame["target"] = target
        target_column = "target"
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    dataset_id = service.register_csv(path)
    scoring_id = None
    if family == "anomaly":
        scoring = frame.iloc[:16].copy()
        scoring.iloc[:4] += 100
        scoring.to_csv(tmp_path / "scoring.csv", index=False)
        scoring_id = service.register_csv(tmp_path / "scoring.csv")
    return service, TaskSpec(family=family, dataset_id=dataset_id, features=features, target=target_column, scoring_dataset_id=scoring_id), frame, target


@pytest.mark.parametrize("family, methods, metric", [
    ("regression", ["ridge", "random_forest_regressor"], "validation_mae"),
    ("classification", ["logistic_regression", "random_forest_classifier"], "validation_balanced_accuracy"),
    ("clustering", ["kmeans", "gaussian_mixture"], "validation_silhouette"),
    ("anomaly", ["isolation_forest", "local_outlier_factor"], None),
])
def test_candidates_produce_compatible_real_evidence(tmp_path, family, methods, metric):
    service, task, frame, target = make_task(tmp_path, family)
    results = [service.run_candidate(task, method, seed=17) for method in methods]
    assert all(result.status == "ok" for result in results), [result.error for result in results]
    comparison = service.compare_results([result.run_id for result in results])
    assert comparison["family"] == family
    for result in results:
        public = service.retrieve_result(result.run_id)
        private = json.loads((service.root / "runs" / result.run_id / "private.json").read_text())
        assert "test_metrics" not in public and "test_predictions" not in public["summary"]
        assert (service.root / "runs" / result.run_id / "model.joblib").is_file()
        if metric:
            assert result.metrics[metric] is not None
        if family == "anomaly":
            assert len(result.summary["ranking"]) == 10
            assert service.evaluation_metrics(result.run_id, {"anomaly_ids": ["0", "1", "2", "3"]})["precision_at_4"] == 1
            assert result.metrics == {}
        elif family == "clustering":
            assert service.evaluation_metrics(result.run_id, {"labels": target.tolist()})["test_adjusted_rand_index"] > 0.8
            saved = joblib.load(service.root / "runs" / result.run_id / "model.joblib")
            validation = private["validation_indices"]
            transformed = saved["preprocessor"].transform(frame[task.features].to_numpy()[validation])
            assignments = saved["estimator"].predict(transformed)
            assert result.metrics[metric] == pytest.approx(silhouette_score(transformed, assignments))
        else:
            assert service.evaluation_metrics(result.run_id)
            if family == "regression":
                assert result.summary["baseline_strategy"] == "training_target_median"


def test_preprocessing_uses_training_rows_only(tmp_path):
    service, task, frame, _ = make_task(tmp_path, "regression")
    frame.loc[::5, "x0"] = np.nan
    frame.to_csv(tmp_path / "missing.csv", index=False)
    task.dataset_id = service.register_csv(tmp_path / "missing.csv")
    result = service.run_candidate(task, "ridge", seed=6)
    assert result.status == "ok", result.error
    private = json.loads((service.root / "runs" / result.run_id / "private.json").read_text())
    expected = np.nanmedian(frame[task.features].to_numpy()[private["training_indices"]], axis=0)
    assert np.allclose(private["imputer_statistics"], expected)
    assert not np.allclose(private["imputer_statistics"], np.nanmedian(frame[task.features].to_numpy(), axis=0))
    partitions = [set(private[name]) for name in ("training_indices", "validation_indices", "test_indices")]
    assert not partitions[0] & partitions[1] and not partitions[0] & partitions[2] and not partitions[1] & partitions[2]


def test_novelty_requires_fresh_data_but_accepts_small_batch(tmp_path):
    service, task, frame, _ = make_task(tmp_path, "anomaly")
    task.scoring_dataset_id = task.dataset_id
    result = service.run_candidate(task, "local_outlier_factor")
    assert result.status == "error" and "fresh" in result.error
    frame.iloc[:1].to_csv(tmp_path / "one.csv", index=False)
    task.scoring_dataset_id = service.register_csv(tmp_path / "one.csv")
    result = service.run_candidate(task, "local_outlier_factor")
    assert result.status == "ok", result.error
    assert result.summary["scoring_rows"] == 1
    saved = joblib.load(service.root / "runs" / result.run_id / "model.joblib")
    assert saved["estimator"].novelty is True


def _delayed_worker(connection, *args):
    time.sleep(10)


def test_timeout_is_retained_as_error_artifact_and_process_stops(tmp_path, monkeypatch):
    import llm_tool_calling_lab.models as models
    monkeypatch.setattr(models, "_worker", _delayed_worker)
    service, task, _, _ = make_task(tmp_path, "regression")
    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    result = service.run_candidate(task, "ridge", timeout_s=0.3)
    assert result.status == "error" and "Timeout" in result.error
    assert time.monotonic() - started < 5
    assert {child.pid for child in multiprocessing.active_children()} <= before
    assert service.retrieve_result(result.run_id)["status"] == "error"
    assert service.evaluation_metrics(result.run_id)["status"] == "error"


def test_unknown_or_wrong_family_methods_rejected(tmp_path):
    service, task, _, _ = make_task(tmp_path, "regression")
    for method in ("invented", "logistic_regression", "../../private"):
        result = service.run_candidate(task, method)
        assert result.status == "error"
    with pytest.raises(ValueError):
        service.retrieve_result("../../private")


def test_tutorial_replacement_exercised(tmp_path):
    service, task, _, _ = make_task(tmp_path, "regression", catalog="tutorial")
    identifiers = {method["method_id"] for method in service.list_methods("regression")}
    assert "elastic_net" in identifiers and "ridge" not in identifiers
    result = service.run_candidate(task, "elastic_net")
    assert result.status == "ok", result.error
    assert result.metrics["validation_mae"] < result.metrics["baseline_validation_mae"]


@pytest.mark.parametrize("family,method", [("regression", "ridge"), ("classification", "logistic_regression"), ("anomaly", "isolation_forest"), ("clustering", "kmeans")])
@pytest.mark.parametrize("defect", ["wrong_length", "matrix", "nonfinite"])
def test_adapter_output_is_validated_before_artifacts_are_published(tmp_path, monkeypatch, family, method, defect):
    import llm_tool_calling_lab.models as models
    service, task, _, _ = make_task(tmp_path, family)
    class MalformedEstimator:
        classes_ = np.array([0, 1])

        def fit(self, *args):
            return self

        def predict(self, matrix):
            if defect == "wrong_length":
                return np.zeros(len(matrix) - 1)
            if defect == "matrix":
                return np.zeros((len(matrix), 1))
            values = np.zeros(len(matrix))
            values[0] = np.inf
            return values

        score_samples = predict

    monkeypatch.setattr(models, "build_estimator", lambda *args: MalformedEstimator())
    run_id = "run_" + "a" * 32
    with pytest.raises(ValueError, match="one-dimensional|finite real"):
        models._execute(str(service.root), "default", task.model_dump(), method, 0, run_id)
    assert not (service.root / "runs" / run_id / "result.json").exists()
    assert not (service.root / "runs" / run_id / "predictions.json").exists()


def test_classifier_adapter_cannot_return_unknown_category_codes(tmp_path, monkeypatch):
    import llm_tool_calling_lab.models as models
    service, task, _, _ = make_task(tmp_path, "classification")
    class InvalidClassifier:
        classes_ = np.array([0, 1])

        def fit(self, *args):
            return self

        def predict(self, matrix):
            return np.full(len(matrix), 2)

    monkeypatch.setattr(models, "build_estimator", lambda *args: InvalidClassifier())
    with pytest.raises(ValueError, match="integer category codes"):
        models._execute(str(service.root), "default", task.model_dump(), "logistic_regression", 0, "run_" + "b" * 32)
