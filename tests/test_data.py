import numpy as np
import pandas as pd
import pytest

from llm_tool_calling_lab.contracts import TaskSpec
from llm_tool_calling_lab.data import DatasetStore, split_indices, validate_task


def register(tmp_path, frame):
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    store = DatasetStore(tmp_path / "state")
    return store, store.register_csv(path)


def valid_frame(size=100):
    return pd.DataFrame({"a": np.arange(size, dtype=float), "b": np.arange(size, dtype=float) ** 2, "target": np.arange(size) % 2})


def test_schema_inspection_never_discloses_values(tmp_path):
    store, dataset_id = register(tmp_path, valid_frame())
    inspection = store.inspect(dataset_id)
    assert inspection["rows"] == 100
    assert set(inspection["columns"][0]) == {"name", "dtype", "numeric", "missing"}
    assert store.register_csv(tmp_path / "input.csv") == dataset_id
    with pytest.raises(ValueError, match="Invalid dataset"):
        store.load("../../other")


@pytest.mark.parametrize("contents", ["a,a\n1,2\n", "a,b\n1,2,3\n", "a,b\n1\n", "", "a,\n1,2\n"])
def test_malformed_csv_rejected(tmp_path, contents):
    path = tmp_path / "bad.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        DatasetStore(tmp_path / "state").register_csv(path)


@pytest.mark.parametrize("defect, expected", [("rows", "100"), ("missing_target", "Missing target"), ("infinite_feature", "Infinite feature"), ("infinite_target", "Infinite target"), ("all_missing", "entirely missing"), ("nonnumeric", "numerical features"), ("three_classes", "exactly two"), ("rare_class", "five rows")])
def test_invalid_tasks_fail_before_fitting(tmp_path, defect, expected):
    frame = valid_frame(99 if defect == "rows" else 100)
    if defect == "missing_target":
        frame.loc[0, "target"] = np.nan
    elif defect == "infinite_feature":
        frame.loc[0, "a"] = np.inf
    elif defect == "infinite_target":
        frame["target"] = frame["target"].astype(float)
        frame.loc[0, "target"] = np.inf
    elif defect == "all_missing":
        frame["a"] = np.nan
    elif defect == "nonnumeric":
        frame["a"] = "text"
    elif defect == "three_classes":
        frame.loc[0, "target"] = 2
    elif defect == "rare_class":
        frame["target"] = 0
        frame.loc[0, "target"] = 1
    store, dataset_id = register(tmp_path, frame)
    with pytest.raises(ValueError, match=expected):
        validate_task(store, TaskSpec(family="classification", dataset_id=dataset_id, features=["a", "b"], target="target"))


def test_target_cannot_be_a_feature(tmp_path):
    store, dataset_id = register(tmp_path, valid_frame())
    with pytest.raises(ValueError, match="target column cannot"):
        validate_task(store, TaskSpec(family="regression", dataset_id=dataset_id, features=["a", "target"], target="target"))


def test_partition_is_disjoint_reproducible_and_stratified():
    labels = np.array([0] * 95 + [1] * 5)
    partitions = split_indices(100, 7, labels)
    assert [len(part) for part in partitions] == [60, 20, 20]
    assert len(set().union(*map(set, partitions))) == 100
    assert all(set(labels[part]) == {0, 1} for part in partitions)
    assert all(np.array_equal(left, right) for left, right in zip(partitions, split_indices(100, 7, labels)))


def test_registered_data_mutation_detected(tmp_path):
    store, dataset_id = register(tmp_path, valid_frame())
    (store.root / f"{dataset_id}.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        store.load(dataset_id)
