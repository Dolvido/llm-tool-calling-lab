from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_evidence.py"
SPEC = importlib.util.spec_from_file_location("export_evidence", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def synthetic_campaign(tmp_path):
    project = tmp_path / "Project with spaces"
    campaign = project / ".lab" / "campaign"
    fixture = campaign / "fixtures" / "c000" / "data.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"x1,x2\r\n1.1,2.2\r\n")
    cases = [{"case_id": "c000", "data": "c000/data.csv", "scoring_data": None,
              "family": "regression", "split": "development", "truth": {"labels": [1]}}]
    write_json(campaign / "fixtures" / "cases.json", cases)
    write_json(campaign / "manifest.json", {"fixture_hashes": {
        relative: sha256((campaign / relative).read_bytes()).hexdigest()
        for relative in ("fixtures/cases.json", "fixtures/c000/data.csv")},
        "repeats": {"development": 1}, "sources": {"models.py": "original-source-hash"}})
    episode = "c000_structured_0"
    run_id = "run_" + "a" * 32
    run = campaign / "artifacts" / episode / "runs" / run_id
    result = {"run_id": run_id, "status": "ok", "metrics": {"validation_mae": 0.123456789},
              "artifact_paths": {"result": str(run / "result.json"), "predictions": str(run / "predictions.json")}}
    write_json(run / "result.json", result)
    write_json(run / "predictions.json", [{"row_id": "1", "prediction": 2.718}])
    write_json(run / "private.json", {"test_metrics": {"test_mae": 0.987}, "test_indices": [1]})
    (run / "model.joblib").write_bytes(b"PRIVATE BINARY DO NOT EXPORT")
    (run / "secret.txt").write_text("DO NOT EXPORT", encoding="utf-8")
    session_name = f"artifacts/{episode}/sessions/{episode}.json"
    write_json(campaign / session_name, {"session_id": episode, "results": [result],
               "events": [{"kind": "tool_failure", "error": "Retained failure"}],
               "messages": [{"role": "assistant", "content": "MAE remains 0.123456789."}]})
    write_json(campaign / "records" / f"{episode}.json", {"episode_id": episode, "case_id": "c000", "condition": "structured",
               "status": "error", "session_file": session_name, "review": {"reviewer_type": "AI", "reviewer": "test reviewer", "rationale": "Fixture test"}})
    write_json(campaign / "records" / "user_data.json", {"secret": "DO NOT EXPORT"})
    write_json(campaign / "artifacts" / episode / "datasets" / "user.json", {"secret": "DO NOT EXPORT"})
    (campaign / ".runner.lock").write_bytes(b"0")
    write_json(campaign / "CAMPAIGN_PROVENANCE.json", {"source_commit": "original-commit", "previous_source": str(project / "src" / "models.py")})
    write_json(campaign / "STARTUP_FAILURE.json", {"status": "error", "reason": "Retained startup failure"})
    (campaign / "REPORT.md").write_text(f"Evidence: {run / 'result.json'}\nSource: {project / 'src' / 'models.py'}\nMAE 0.123456789\n", encoding="utf-8")
    return campaign, project, run_id


def test_export_preserves_csv_metrics_failure_and_hash_provenance(synthetic_campaign, tmp_path):
    campaign, project, run_id = synthetic_campaign
    destination = tmp_path / "release.zip"
    before = {path: path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
    receipt = exporter.export_evidence(campaign, destination)
    assert receipt["records"] == 1 and receipt["failed_records"] == 1
    assert receipt["sha256"] == sha256(destination.read_bytes()).hexdigest()
    assert Path(receipt["sha256_file"]).read_text().startswith(receipt["sha256"])
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert not any(name.endswith((".joblib", ".lock", ".tmp", "secret.txt", "user_data.json")) or "/datasets/" in name for name in names)
        assert "fixtures/cases.json" in names
        assert "CAMPAIGN_PROVENANCE.json" in names and "STARTUP_FAILURE.json" in names
        assert archive.read("fixtures/c000/data.csv") == before[campaign / "fixtures/c000/data.csv"]
        result_name = f"artifacts/c000_structured_0/runs/{run_id}/result.json"
        result = json.loads(archive.read(result_name))
        assert result["metrics"]["validation_mae"] == 0.123456789
        assert result["artifact_paths"]["result"] == result_name
        index = json.loads(archive.read("export_index.json"))
        item = next(item for item in index["files"] if item["path"] == result_name)
        assert item["original_sha256"] == sha256(before[campaign / result_name]).hexdigest()
        assert item["exported_sha256"] == sha256(archive.read(result_name)).hexdigest()
        assert item["sanitized"] is True
        assert index["review_provenance"] == [{"type": "AI", "reviewer": "test reviewer"}]
        assert index["counts"]["required_expected"] == 3
        assert index["counts"]["required_records"] == 1
        assert index["counts"]["required_missing"] == 2
        joined = b"\n".join(archive.read(name) for name in names)
        assert str(project).encode() not in joined and project.as_posix().encode() not in joined
        assert b"PRIVATE BINARY" not in joined and b"DO NOT EXPORT" not in joined
        assert b"Retained failure" in joined and b"MAE remains 0.123456789" in joined
        assert b"evaluator-only ground truth" in archive.read("README.md")
    assert all(path.read_bytes() == content for path, content in before.items())


def test_export_rejects_changed_fixture(synthetic_campaign, tmp_path):
    campaign, _, _ = synthetic_campaign
    (campaign / "fixtures/c000/data.csv").write_bytes(b"USER DATA")
    with pytest.raises(ValueError, match="Frozen fixture"):
        exporter.export_evidence(campaign, tmp_path / "release.zip")
    assert not (tmp_path / "release.zip").exists()


def test_export_rejects_running_record_and_write_inside_campaign(synthetic_campaign, tmp_path):
    campaign, _, _ = synthetic_campaign
    with pytest.raises(ValueError, match="outside"):
        exporter.export_evidence(campaign, campaign / "release.zip")
    path = campaign / "records/c000_structured_0.json"
    record = json.loads(path.read_text())
    record["status"] = "running"
    write_json(path, record)
    with pytest.raises(ValueError, match="not completed"):
        exporter.export_evidence(campaign, tmp_path / "release.zip")


def test_export_rejects_unknown_absolute_paths(synthetic_campaign, tmp_path):
    campaign, _, _ = synthetic_campaign
    (campaign / "REPORT.md").write_text("Unrelated source C:\\Private\\secret.csv", encoding="utf-8")
    with pytest.raises(ValueError, match="absolute local path"):
        exporter.export_evidence(campaign, tmp_path / "release.zip")


def test_export_rejects_fixture_path_traversal(synthetic_campaign, tmp_path):
    campaign, _, _ = synthetic_campaign
    path = campaign / "fixtures/cases.json"
    cases = json.loads(path.read_text())
    cases[0]["data"] = "../../private.csv"
    write_json(path, cases)
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fixture_hashes"]["fixtures/cases.json"] = sha256(path.read_bytes()).hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Noncanonical"):
        exporter.export_evidence(campaign, tmp_path / "release.zip")
