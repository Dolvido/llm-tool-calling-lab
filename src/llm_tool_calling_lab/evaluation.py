"""Frozen campaigns, append-preserving execution, and transparent scoring."""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
import json
import os
import statistics
import sys

from .contracts import LabConfig, TaskSpec
from .fixtures import FAMILIES, generate_examples

REFERENCES = {"regression": "ridge", "classification": "logistic_regression", "anomaly": "isolation_forest", "clustering": "kmeans"}
RUBRIC = {
    "version": "0.1.0",
    "supported": ["appropriate_task_target", "valid_required_output", "faithful_claims", "relevant_followup", "appropriate_limits"],
    "challenge": ["specific_limitation_or_clarification", "appropriate_next_step", "faithful_claims"],
    "review_policy": "Qualitative scores require an identified reviewer and rationale. AI review must be labeled AI, never human. Unreviewed is not a pass.",
}

def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    temporary.replace(path)

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()

def source_fingerprints():
    folder = Path(__file__).parent
    names = ("contracts.py", "backend.py", "chat.py", "catalog.py", "data.py", "models.py", "fixtures.py", "evaluation.py")
    return {name: _digest(folder / name) for name in names}

def package_versions():
    return dict(sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata.get("Name")))

def backend_identity(config):
    import ollama
    client = ollama.Client(host=config.host, timeout=20)
    models = client.list().model_dump().get("models", [])
    model = next((m for m in models if m.get("model") == config.model), None)
    if not model:
        raise ValueError(f"Configured model is not installed: {config.model}")
    return {"model": config.model, "digest": model["digest"], "size": model.get("size")}

def freeze(destination: Path, config: LabConfig, families=FAMILIES, identity=None, seed_offset: int = 0) -> Path:
    destination = Path(destination).resolve()
    if getattr(config, "catalog", "default") != "default":
        raise ValueError("Benchmark campaigns require the default catalog; the tutorial is evaluated separately")
    if (destination / "manifest.json").exists():
        raise ValueError("Campaign already frozen; use a new destination for another version")
    if len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise ValueError("Invalid or duplicate family")
    cases = generate_examples(destination / "fixtures", families, seed_offset=seed_offset)
    fixture_hashes = {str(p.relative_to(destination)).replace("\\", "/"): _digest(p)
                      for p in sorted((destination / "fixtures").rglob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "families": list(families), "config": config.model_dump(), "seed_offset": seed_offset,
                "backend": identity or backend_identity(config), "python": sys.version,
                "packages": package_versions(), "sources": source_fingerprints(),
                "fixture_hashes": fixture_hashes, "rubric": RUBRIC,
                "counts": {"development": 4*len(families), "heldout": 24*len(families), "challenge": 24},
                "reference_counts": {"development": 2*len(families), "heldout": 6*len(families)},
                "repeats": {"development": 1, "heldout": 2, "challenge": 2, "challenge_development": 1}}
    _write(destination / "manifest.json", manifest)
    return destination / "manifest.json"

def verify_freeze(destination: Path, check_backend=True):
    destination = Path(destination).resolve()
    manifest = _read(destination / "manifest.json")
    if source_fingerprints() != manifest["sources"]:
        raise ValueError("Evaluated source changed after freeze; create a new campaign, preserving the old one")
    if package_versions() != manifest["packages"]:
        raise ValueError("Environment changed after freeze")
    for name, expected in manifest["fixture_hashes"].items():
        if _digest(destination / name) != expected:
            raise ValueError(f"Frozen fixture changed: {name}")
    if check_backend and backend_identity(LabConfig(**manifest["config"])) != manifest["backend"]:
        raise ValueError("Backend identity changed after freeze")
    return manifest

@contextmanager
def _campaign_lock(destination):
    # OS file locks release on process death and protect resume from duplicate writers.
    path = Path(destination).resolve() / ".runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)

def _task(case, dataset_id, scoring_id):
    return TaskSpec(question=case["question"], family=case["family"], dataset_id=dataset_id,
                    features=case["features"], target=case.get("target"),
                    scoring_dataset_id=scoring_id, n_clusters=case.get("n_clusters", 3))

def _machine_checks(case, responses, service, session, dataset_id=None, scoring_id=None):
    valid = len(responses) == 2 and all(r.get("status") == "ok" for r in responses)
    errors = []
    if not valid:
        errors.append("incomplete_or_failed_conversation")
    if case["family"] != "challenge":
        selected = responses[-1].get("selected_run_id") if responses else None
        if not selected:
            errors.append("no_selected_artifact")
        else:
            try:
                artifact = service.retrieve_result(selected)
                task = artifact["task"]
                if artifact["status"] != "ok" or task["family"] != case["family"] or task.get("target") != case.get("target"):
                    errors.append("wrong_task_or_failed_artifact")
                if sorted(task.get("features", [])) != sorted(case.get("features", [])):
                    errors.append("wrong_feature_columns")
                if case["family"] == "clustering" and task.get("n_clusters") != case.get("n_clusters", 3):
                    errors.append("wrong_group_count")
                if dataset_id is not None and task.get("dataset_id") != dataset_id:
                    errors.append("wrong_fitting_dataset")
                if task.get("scoring_dataset_id") != scoring_id:
                    errors.append("wrong_scoring_dataset")
                if case["family"] == "anomaly":
                    output_ids = responses[-1].get("row_ids", [])
                    ranked_ids = [str(r["row_id"]) for r in artifact.get("summary", {}).get("ranking", [])[:4]]
                    if len(output_ids) != 4 or len(set(output_ids)) != 4 or output_ids != ranked_ids:
                        errors.append("invalid_anomaly_answer_ids")
                for response in responses:
                    for item in response.get("evidence", []):
                        saved = service.retrieve_result(item["run_id"])
                        if item.get("metrics") != saved.get("metrics") or item.get("summary") != saved.get("summary"):
                            errors.append("evidence_value_mismatch")
            except Exception:
                errors.append("invalid_artifact_reference")
    state = session.state
    for key, cap in (("fits", session.config.max_fits), ("tool_calls", session.config.max_tool_calls),
                     ("llm_responses", session.config.max_llm_responses), ("completion_tokens", session.config.max_episode_tokens),
                     ("repairs", session.config.max_repairs), ("active_seconds", session.config.max_seconds)):
        if state[key] > cap:
            errors.append("budget_exceeded_" + key)
    return {"pass": not errors, "errors": errors,
            "note": "Structural checks only. Prose fidelity, exact required IDs, and interpretation require rubric review."}

def run_campaign(destination: Path, split="development", limit=None, backend_factory=None):
    """Resume by episode ID; failed/interrupted attempts remain, never silently rerun."""
    from .models import ModelService
    from .chat import Session
    destination = Path(destination).resolve()
    with _campaign_lock(destination):
        manifest = verify_freeze(destination, check_backend=backend_factory is None)
        if split not in manifest["repeats"]:
            raise ValueError("Unknown split")
        config = LabConfig(**manifest["config"])
        cases = [c for c in _read(destination / "fixtures" / "cases.json") if c["split"] == split]
        completed = 0
        for case in cases:
            for condition in ("reference", "generic", "structured"):
                if condition == "reference" and case["family"] == "challenge":
                    continue
                repeats = 1 if condition == "reference" else manifest["repeats"][split]
                for repeat in range(repeats):
                    identifier = f"{case['case_id']}_{condition}_{repeat}"
                    path = destination / "records" / (identifier + ".json")
                    if path.exists():
                        old = _read(path)
                        if old["status"] == "running":
                            old.update(status="error", error="interrupted_attempt_preserved", machine={"pass": False, "errors": ["interrupted"]})
                            if old.get("family") == "anomaly":
                                old["model_metrics"] = {"precision_at_4": 0.0}
                            _write(path, old)
                        continue
                    if limit is not None and completed >= limit:
                        return completed
                    record = {"episode_id": identifier, "case_id": case["case_id"], "split": split,
                              "family": case["family"], "condition": condition, "repeat": repeat,
                              "status": "running", "responses": [], "review": None,
                              "expected": case["expected"], "model_metrics": {}, "machine": {"pass": False, "errors": ["not_completed"]}}
                    _write(path, record)
                    try:
                        service = ModelService(destination / "artifacts" / identifier)
                        dataset_id = service.register_csv(destination / "fixtures" / case["data"])
                        scoring_id = service.register_csv(destination / "fixtures" / case["scoring_data"]) if case["scoring_data"] else None
                        if condition == "reference":
                            result = service.run_candidate(_task(case, dataset_id, scoring_id), REFERENCES[case["family"]], seed=config.seed)
                            record["result"] = result.model_dump()
                            record["status"] = result.status
                            record["model_metrics"] = service.evaluation_metrics(result.run_id, case["truth"]) if result.status == "ok" else {}
                            record["machine"] = {"pass": result.status == "ok", "errors": [] if result.status == "ok" else [result.error]}
                        else:
                            backend = backend_factory(config, case, condition) if backend_factory else None
                            session = Session(service, config, condition=condition, backend=backend, session_id=identifier)
                            record["session_file"] = str(session.path.relative_to(destination)).replace("\\", "/")
                            for question in (case["question"], case["followup"]):
                                response = session.respond(question, dataset_id, scoring_id)
                                record["responses"].append(response)
                                _write(path, record)
                            record["usage"] = session.state
                            record["machine"] = _machine_checks(case, record["responses"], service, session, dataset_id, scoring_id)
                            record["status"] = "ok" if record["machine"]["pass"] else "error"
                            selected = record["responses"][-1].get("selected_run_id")
                            if selected and case["family"] != "challenge":
                                record["model_metrics"] = service.evaluation_metrics(selected, case["truth"])
                            if case["family"] == "anomaly":
                                answer_ids = record["responses"][-1].get("row_ids", [])
                                truth_ids = set(case["truth"].get("anomaly_ids", []))
                                record["model_metrics"]["precision_at_4"] = (len(set(answer_ids) & truth_ids) / 4
                                    if record["machine"]["pass"] else 0.0)
                    except Exception as exc:
                        record.update(status="error", error=f"{type(exc).__name__}: {exc}", machine={"pass": False, "errors": ["execution_error"]})
                        if case["family"] == "anomaly":
                            record["model_metrics"] = {"precision_at_4": 0.0}
                    _write(path, record)
                    completed += 1
                    print(f"{identifier}: {record['status']}", flush=True)
        return completed

def review_episode(destination: Path, episode_id: str, checks: dict, reviewer: str, reviewer_type: str, rationale: str):
    if reviewer_type not in {"human", "AI"} or not reviewer.strip() or not rationale.strip():
        raise ValueError("Review requires identity, human/AI label, and rationale")
    if not episode_id.replace("_", "").isalnum():
        raise ValueError("Invalid episode ID")
    path = Path(destination).resolve() / "records" / (episode_id + ".json")
    record = _read(path)
    required = RUBRIC["challenge" if record["family"] == "challenge" else "supported"]
    if set(checks) != set(required) or not all(type(v) is bool for v in checks.values()):
        raise ValueError(f"Review requires boolean checks: {required}")
    review = {"reviewer": reviewer, "reviewer_type": reviewer_type, "rubric_version": RUBRIC["version"],
              "checks": checks, "rationale": rationale, "timestamp": datetime.now(timezone.utc).isoformat()}
    # Reviews are append-only amendments, preserving prior scoring history.
    record.setdefault("review_history", []).append(review)
    record["review"] = review
    _write(path, record)

def completion(record):
    if not record.get("machine", {}).get("pass"):
        return 0
    review = record.get("review")
    if review is None:
        return None
    return int(all(review["checks"].values()))


def _expected_groups(destination, manifest):
    """Initial protocol denominators, excluding optional challenge development runs."""
    expected = defaultdict(int)
    cases = _read(Path(destination) / "fixtures" / "cases.json")
    for case in cases:
        if case["split"] == "challenge_development":
            continue
        for condition in ("reference", "generic", "structured"):
            if condition == "reference" and case["family"] == "challenge":
                continue
            repeats = 1 if condition == "reference" else manifest["repeats"][case["split"]]
            expected[(case["split"], case["family"], condition)] += repeats
    return expected


def _paired_completion(records, manifest):
    """Average repeated completion scores within each case before pairing arms."""
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record["condition"] in {"generic", "structured"}:
            groups[(record["split"], record["family"], record["case_id"])][record["condition"]].append(record)
    outcomes = defaultdict(lambda: dict(structured_wins=0, generic_wins=0, ties=0,
                                       both_fail=0, pending_review=0, incomplete=0))
    for (split, family, _), arms in groups.items():
        row = outcomes[(split, family)]
        repeats = manifest["repeats"][split]
        if any(len(arms[arm]) != repeats for arm in ("generic", "structured")):
            row["incomplete"] += 1
            continue
        scores = {arm: [completion(r) for r in arms[arm]] for arm in ("generic", "structured")}
        if any(score is None for values in scores.values() for score in values):
            row["pending_review"] += 1
            continue
        generic, structured = (statistics.mean(scores[arm]) for arm in ("generic", "structured"))
        category = ("both_fail" if generic == structured == 0 else
                    "ties" if generic == structured else
                    "structured_wins" if structured > generic else "generic_wins")
        row[category] += 1
    return [{"split": split, "family": family, **values} for (split, family), values in sorted(outcomes.items())]


def write_report(destination: Path) -> Path:
    destination = Path(destination).resolve()
    manifest = _read(destination / "manifest.json")
    records = [_read(p) for p in sorted((destination / "records").glob("*.json"))]
    groups = defaultdict(list)
    for record in records:
        groups[(record["split"], record["family"], record["condition"])].append(record)
    expected = _expected_groups(destination, manifest)
    required_total = sum(expected.values())
    required_recorded = sum(len(rows) for key, rows in groups.items() if key in expected)
    required_missing = sum(max(0, count - len(groups[key])) for key, count in expected.items())
    lines = ["# Evaluation results", "", "Experimental synthetic-data evaluation. Repeats are not independent datasets.", "",
             f"Model: `{manifest['backend']['model']}`. Families frozen: {', '.join(manifest['families'])}.", "",
             f"Initial protocol: {required_recorded}/{required_total} episode records, including deterministic references; **{required_missing} required episodes missing**. Optional challenge-development runs are additional and do not replace required episodes.", "",
             "Missing qualitative reviews remain unscored. AI review, if present, is identified and is not an independent human study.", "",
             "| Split | Family | Condition | Expected | Recorded | Missing | Structural passes | Primary passes | Pending review |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(set(expected) | set(groups)):
        rows = groups[key]
        scores = [completion(r) for r in rows] if key[2] != "reference" else []
        primary = str(sum(s == 1 for s in scores)) if scores else "N/A"
        count = expected.get(key, len(rows))
        lines.append(f"| {' | '.join(key)} | {count} | {len(rows)} | {max(0, count-len(rows))} | {sum(r.get('machine',{}).get('pass',False) for r in rows)} | {primary} | {sum(s is None for s in scores)} |")
    paired = _paired_completion(records, manifest)
    lines += ["", "## Paired completion by case", "", "Repeated primary-completion scores are averaged inside each case before comparing conditions. A failed episode scores zero; unreviewed episodes remain pending. Cases missing either arm or repeat are incomplete. Both-zero cases are shown separately from ties. No significance claim is made.", "",
              "| Split | Family | Structured wins | Generic wins | Ties | Both fail | Pending review | Incomplete |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['split']} | {row['family']} | " + " | ".join(str(row[k]) for k in ("structured_wins", "generic_wins", "ties", "both_fail", "pending_review", "incomplete")) + " |")
    lines += ["", "## Model quality", "", "Metrics are averaged within a case before averaging cases. Missing metrics are unavailable, not silently treated as successes. Inspect failures above and episode records.", "",
              "| Split | Family | Condition | Metric | Case mean | Cases with metric |", "|---|---|---|---|---:|---:|"]
    summaries = []
    for key, rows in sorted(groups.items()):
        metrics = sorted({m for row in rows for m,v in row.get("model_metrics",{}).items() if isinstance(v,(int,float))})
        for metric in metrics:
            bycase = defaultdict(list)
            for row in rows:
                value = row.get("model_metrics",{}).get(metric)
                if isinstance(value,(int,float)):
                    bycase[row["case_id"]].append(value)
            values = [statistics.mean(v) for v in bycase.values()]
            mean = statistics.mean(values) if values else None
            summaries.append({"split":key[0],"family":key[1],"condition":key[2],"metric":metric,"case_mean":mean,"cases":len(values)})
            lines.append(f"| {' | '.join(key)} | {metric} | {mean:.4f} | {len(values)} |")
    lines += ["", "## Usage and failures", "", "Local inference incurred no paid API calls; electricity and hardware costs were not measured.", ""]
    for condition in ("generic", "structured"):
        subset = [r for r in records if r["condition"] == condition]
        seconds = [r["usage"]["active_seconds"] for r in subset if isinstance(r.get("usage",{}).get("active_seconds"), (int, float))]
        latency = f"{statistics.mean(seconds):.2f}" if seconds else "unavailable"
        lines.append(f"- {condition}: {len(subset)} recorded episodes; mean active seconds {latency} ({len(seconds)} measured, {len(subset)-len(seconds)} unavailable); {sum(r['status']=='error' for r in subset)} structural/execution failures.")
    lines += ["", "## Review provenance", ""]
    reviewers = sorted({(r["review"]["reviewer_type"],r["review"]["reviewer"]) for r in records if r.get("review")})
    lines += [f"- {kind}: {who}" for kind,who in reviewers] or ["No qualitative reviews recorded."]
    lines += ["", "## Limits", "", "These results concern small synthetic numerical tables and this frozen backend. They do not establish production reliability, causal inference, general intelligence, or reduced human expertise. Failed episodes remain in records and denominators. No superiority claim is inferred from this report.", ""]
    path = destination / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    _write(destination / "summary.json", {"manifest": manifest, "model_quality": summaries,
        "recorded": len(records), "required_total": required_total, "required_recorded": required_recorded,
        "required_missing": required_missing, "paired_completion": paired,
        "qualitative_pending": sum(completion(r) is None for r in records if r["condition"] != "reference"),
        "records": [{"episode_id":r["episode_id"],"status":r["status"],"completion":completion(r)} for r in records]})
    return path
