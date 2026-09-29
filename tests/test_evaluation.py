"""Protocol integrity checks use fake inference only, never held-out model calls."""
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from llm_tool_calling_lab import evaluation as ev
from llm_tool_calling_lab.contracts import LabConfig, RunResult
from llm_tool_calling_lab.fixtures import FAMILIES, generate_examples


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "package_versions", lambda: {"test-runtime": "1"})
    monkeypatch.setattr(ev, "backend_identity", lambda config: {"model": config.model, "digest": "fixed-test-digest"})
    ev.freeze(tmp_path, LabConfig())
    return tmp_path


def test_fixture_counts_independent_seeds_and_truth_not_in_csv(tmp_path):
    cases = generate_examples(tmp_path)
    assert Counter(c["split"] for c in cases) == {
        "development": 8, "heldout": 24, "challenge": 6, "challenge_development": 6}
    seeds = [{c["seed"] for c in cases if c["split"] == split} for split in ("development", "heldout")]
    assert not seeds[0] & seeds[1]
    for case in cases:
        data = pd.read_csv(tmp_path / case["data"])
        assert not {"anomaly_ids", "labels", "hidden_labels", "expected", "generator"} & set(data.columns)
        if case["family"] == "anomaly":
            assert len(case["truth"]["anomaly_ids"]) == 4
            assert len(pd.read_csv(tmp_path / case["scoring_data"])) == 64
        if case["family"] == "clustering":
            assert len(case["truth"]["labels"]) == len(data)
    analogues = [c for c in cases if c["split"] == "challenge_development"]
    challenges = [c for c in cases if c["split"] == "challenge"]
    assert all(a["question"] != b["question"] and a["followup"] != b["followup"]
               for a, b in zip(analogues, challenges))
    assert len({c["question"] for c in cases if c["family"] == "regression"}) > 1


def test_frozen_denominators_and_reference_counts(campaign):
    manifest = ev.verify_freeze(campaign)
    assert manifest["counts"] == {"development": 16, "heldout": 96, "challenge": 24}
    assert manifest["reference_counts"] == {"development": 8, "heldout": 24}
    assert sum(ev._expected_groups(campaign, manifest).values()) == 168


def test_same_fixture_generation_is_reproducible(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    generate_examples(first)
    generate_examples(second)
    files = [p.relative_to(first) for p in first.rglob("*") if p.is_file()]
    assert all((first / p).read_bytes() == (second / p).read_bytes() for p in files)


def test_freeze_cannot_be_overwritten(campaign):
    before = (campaign / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already frozen"):
        ev.freeze(campaign, LabConfig())
    assert before == (campaign / "manifest.json").read_bytes()


def test_tutorial_catalog_cannot_enter_benchmark(tmp_path):
    config = LabConfig().model_copy(update={"catalog": "tutorial"})
    with pytest.raises(ValueError, match="default catalog"):
        ev.freeze(tmp_path, config)


@pytest.mark.parametrize("change,match", [("source", "source changed"), ("package", "Environment changed"),
                                         ("backend", "Backend identity"), ("fixture", "fixture changed")])
def test_freeze_rejects_source_environment_backend_and_data_changes(campaign, monkeypatch, change, match):
    if change == "source":
        monkeypatch.setattr(ev, "source_fingerprints", lambda: {"chat.py": "changed"})
    elif change == "package":
        monkeypatch.setattr(ev, "package_versions", lambda: {"test-runtime": "2"})
    elif change == "backend":
        monkeypatch.setattr(ev, "backend_identity", lambda config: {"model": config.model, "digest": "changed"})
    else:
        path = campaign / "fixtures" / "c000" / "data.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=match):
        ev.verify_freeze(campaign)


class ReferenceService:
    """Small fake avoids model fits; real Session still checks its tool boundary."""
    def __init__(self, root):
        self.root = Path(root)
        self.runs = {}
    def register_csv(self, path):
        return "ds_example"
    def inspect_dataset(self, dataset_id):
        return {"columns": [{"name": "x1", "numeric": True}, {"name": "x2", "numeric": True}]}
    def list_methods(self, family=None):
        return [{"method_id": "ridge", "family": "regression"}]
    def run_candidate(self, task, method_id, **kwargs):
        result = RunResult(run_id="run_fixed", method_id=method_id, task=task, status="ok")
        self.runs[result.run_id] = result.model_dump(mode="json")
        return result
    def retrieve_result(self, run_id):
        return self.runs[run_id]
    def evaluation_metrics(self, run_id, truth):
        return {}


class ClarificationBackend:
    requests = []
    def chat(self, messages, tools, **kwargs):
        self.requests.append(json.dumps(messages))
        return {"content": json.dumps({"answer": "Please clarify the target.",
                                       "evidence_ids": [], "selected_run_id": None,
                                       "next_action": "Specify the target."}),
                "completion_tokens": 25, "prompt_tokens": 100}


def fake_backend(config, case, condition):
    return ClarificationBackend()


def test_resume_preserves_failures_and_interrupted_records(campaign, monkeypatch):
    monkeypatch.setattr("llm_tool_calling_lab.models.ModelService", ReferenceService)
    assert ev.run_campaign(campaign, "development", limit=3, backend_factory=fake_backend) == 3
    generic = campaign / "records" / "c000_generic_0.json"
    old = generic.read_bytes()
    assert json.loads(old)["status"] == "error"  # No artifact is a real failure.
    # Simulate process death after its first persistent receipt.
    interrupted_path = campaign / "records" / "c001_reference_0.json"
    ev._write(interrupted_path, {"status": "running", "episode_id": "c001_reference_0"})
    assert ev.run_campaign(campaign, "development", limit=0, backend_factory=fake_backend) == 0
    assert generic.read_bytes() == old
    interrupted = ev._read(interrupted_path)
    assert interrupted["status"] == "error"
    assert interrupted["error"] == "interrupted_attempt_preserved"


def test_development_runner_creates_16_llm_and_8_reference_receipts(campaign, monkeypatch):
    monkeypatch.setattr("llm_tool_calling_lab.models.ModelService", ReferenceService)
    ClarificationBackend.requests = []
    assert ev.run_campaign(campaign, "development", backend_factory=fake_backend) == 24
    records = [ev._read(p) for p in (campaign / "records").glob("*.json")]
    assert Counter(r["condition"] for r in records) == {"reference": 8, "generic": 8, "structured": 8}
    assert ev.run_campaign(campaign, "development", backend_factory=fake_backend) == 0
    # Backend contexts contain no evaluator fixture record or truth.
    assert all('"truth"' not in text and '"anomaly_ids"' not in text and '"labels"' not in text
               for text in ClarificationBackend.requests)


def test_interrupted_anomaly_resume_preserves_failure_and_zeroes_answer_metric(campaign, monkeypatch):
    monkeypatch.setattr("llm_tool_calling_lab.models.ModelService", ReferenceService)
    ev.run_campaign(campaign, "development", backend_factory=fake_backend)
    path = campaign / "records" / "c020_generic_0.json"
    interrupted = ev._read(path)
    interrupted.update(status="running", model_metrics={"precision_at_4": 1.0})
    ev._write(path, interrupted)
    assert ev.run_campaign(campaign, "development", limit=0, backend_factory=fake_backend) == 0
    preserved = ev._read(path)
    assert preserved["status"] == "error"
    assert preserved["model_metrics"] == {"precision_at_4": 0.0}


def test_report_shows_missing_pending_unknown_latency_and_all_families(campaign):
    ev._write(campaign / "records" / "c000_structured_0.json", {
        "episode_id": "c000_structured_0", "case_id": "c000", "split": "development",
        "family": "regression", "condition": "structured", "repeat": 0,
        "status": "ok", "machine": {"pass": True}, "review": None, "model_metrics": {}})
    report = ev.write_report(campaign).read_text(encoding="utf-8")
    summary = ev._read(campaign / "summary.json")
    assert summary["required_total"] == 168 and summary["required_missing"] == 167
    assert summary["qualitative_pending"] == 1
    assert "mean active seconds unavailable" in report
    assert all(family in report for family in FAMILIES)
    assert "167 required episodes missing" in report


def test_primary_is_not_a_pass_before_qualitative_review():
    assert ev.completion({"machine": {"pass": True}, "review": None}) is None
    assert ev.completion({"machine": {"pass": False}, "review": None}) == 0


def test_paired_scores_average_repeats_and_keep_both_fail_separate():
    records = []
    for case, generic, structured in (("a", [1, 0], [1, 1]), ("b", [0, 0], [0, 0]),
                                       ("c", [1, 1], [1, 1]), ("d", [None, 1], [1, 1])):
        for arm, values in (("generic", generic), ("structured", structured)):
            for repeat, score in enumerate(values):
                records.append({"split": "heldout", "family": "regression", "case_id": case,
                                "condition": arm, "repeat": repeat, "machine": {"pass": score != 0},
                                "review": None if score is None else {"checks": {"all": bool(score)}}})
    rows = ev._paired_completion(records, {"repeats": {"heldout": 2}})
    assert rows == [{"split": "heldout", "family": "regression", "structured_wins": 1,
                     "generic_wins": 0, "ties": 1, "both_fail": 1, "pending_review": 1, "incomplete": 0}]


def test_reviews_require_explicit_provenance_and_preserve_history(campaign):
    identifier = "c000_structured_0"
    ev._write(campaign / "records" / (identifier + ".json"), {
        "episode_id": identifier, "family": "regression", "machine": {"pass": True}, "review": None})
    checks = dict.fromkeys(ev.RUBRIC["supported"], True)
    with pytest.raises(ValueError, match="identity"):
        ev.review_episode(campaign, identifier, checks, "", "AI", "reason")
    ev.review_episode(campaign, identifier, checks, "test-reviewer", "AI", "Fixture-only review")
    ev.review_episode(campaign, identifier, dict.fromkeys(checks, False), "test-reviewer", "AI", "Revised fixture review")
    record = ev._read(campaign / "records" / (identifier + ".json"))
    assert len(record["review_history"]) == 2 and record["review"]["reviewer_type"] == "AI"
    assert ev.completion(record) == 0


def test_machine_checks_include_time_and_repair_caps():
    session = SimpleNamespace(config=LabConfig(), state={"fits": 0, "tool_calls": 0,
        "llm_responses": 0, "completion_tokens": 0, "repairs": 2, "active_seconds": 181})
    result = ev._machine_checks({"family": "challenge"}, [{"status": "ok"}, {"status": "ok"}], None, session)
    assert set(result["errors"]) == {"budget_exceeded_repairs", "budget_exceeded_active_seconds"}


def test_machine_checks_reject_wrong_features_groups_and_dataset_roles():
    session = SimpleNamespace(config=LabConfig(), state={"fits": 1, "tool_calls": 1,
        "llm_responses": 2, "completion_tokens": 10, "repairs": 0, "active_seconds": 1})
    artifact = {"status": "ok", "task": {"family": "clustering", "target": None,
        "features": ["x2", "x3"], "n_clusters": 4, "dataset_id": "foreign",
        "scoring_dataset_id": "unexpected"}}
    service = SimpleNamespace(retrieve_result=lambda identifier: artifact)
    case = {"family": "clustering", "features": ["x1", "x2"], "n_clusters": 3}
    response = {"status": "ok", "selected_run_id": "run_real", "evidence": []}
    result = ev._machine_checks(case, [response, response], service, session, "expected", None)
    assert set(result["errors"]) == {"wrong_feature_columns", "wrong_group_count", "wrong_fitting_dataset", "wrong_scoring_dataset"}


def test_anomaly_primary_requires_four_actual_answer_ids():
    session = SimpleNamespace(config=LabConfig(), state={"fits": 1, "tool_calls": 1,
        "llm_responses": 2, "completion_tokens": 10, "repairs": 0, "active_seconds": 1})
    artifact = {"status": "ok", "task": {"family": "anomaly", "target": None,
        "features": ["x1", "x2"], "dataset_id": "reference", "scoring_dataset_id": "batch"},
        "summary": {"ranking": [{"row_id": str(i)} for i in range(4)]}}
    service = SimpleNamespace(retrieve_result=lambda identifier: artifact)
    case = {"family": "anomaly", "features": ["x1", "x2"]}
    response = {"status": "ok", "selected_run_id": "run_real", "evidence": [], "row_ids": ["0", "1", "2", "99"]}
    result = ev._machine_checks(case, [response, response], service, session, "reference", "batch")
    assert result["errors"] == ["invalid_anomaly_answer_ids"]
