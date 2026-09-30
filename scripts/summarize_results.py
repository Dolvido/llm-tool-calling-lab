"""Derive public aggregate tables without changing frozen campaign evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

from llm_tool_calling_lab.evaluation import completion


USAGE_FIELDS = {
    "active_seconds": "Active seconds",
    "fits": "Candidate fits",
    "tool_calls": "Tool calls",
    "llm_responses": "LLM responses",
    "completion_tokens": "Accounted generated tokens",
    "prompt_tokens": "Reported prompt/input tokens",
    "repairs": "Repairs",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_results(campaign: Path, destination: Path) -> dict:
    campaign = Path(campaign).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination == campaign or destination.is_relative_to(campaign):
        raise ValueError("Write derived reports outside the frozen campaign")
    manifest = _read(campaign / "manifest.json")
    cases = _read(campaign / "fixtures" / "cases.json")
    expected = {}
    for case in cases:
        for condition in ("reference", "generic", "structured"):
            if condition == "reference" and case["family"] == "challenge":
                continue
            repeats = 1 if condition == "reference" else manifest["repeats"][case["split"]]
            for repeat in range(repeats):
                identifier = f"{case['case_id']}_{condition}_{repeat}"
                expected[identifier] = {"case_id": case["case_id"], "split": case["split"],
                                        "family": case["family"], "condition": condition,
                                        "repeat": repeat}
    records, running, hashes = {}, [], {}
    for path in sorted((campaign / "records").glob("*.json")):
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
        identifier = record["episode_id"]
        if identifier not in expected or path.stem != identifier:
            raise ValueError(f"Unexpected record identity: {path.name}")
        if any(record.get(key) != value for key, value in expected[identifier].items()):
            raise ValueError(f"Record does not match the frozen assignment: {identifier}")
        hashes[identifier] = sha256(raw).hexdigest()
        if record["status"] == "running":
            running.append(identifier)
        elif record["status"] in {"ok", "error"}:
            records[identifier] = record
        else:
            raise ValueError(f"Unknown record status: {identifier}")
    required = {identifier for identifier, item in expected.items()
                if item["split"] != "challenge_development"}
    missing = sorted(required - set(records) - set(running))
    required_running = sorted(required & set(running))
    required_complete = len(required & set(records))
    grouped = defaultdict(list)
    expected_groups = Counter()
    for identifier, item in expected.items():
        key = (item["split"], item["family"], item["condition"])
        if identifier in required or identifier in records or identifier in running:
            expected_groups[key] += 1
        if identifier in records:
            grouped[key].append(records[identifier])
    aggregates, diagnostics = [], []
    for (split, family, condition), denominator in sorted(expected_groups.items()):
        rows = grouped[(split, family, condition)]
        scores = [completion(row) for row in rows] if condition != "reference" else []
        machine_failures = sum(not row.get("machine", {}).get("pass", False) for row in rows)
        aggregates.append({"split": split, "family": family, "condition": condition,
                           "expected": denominator, "completed": len(rows),
                           "not_completed": denominator - len(rows),
                           "structural_passes": len(rows) - machine_failures,
                           "machine_failures": machine_failures,
                           "primary_passes": sum(score == 1 for score in scores) if condition != "reference" else None,
                           "primary_failures": sum(score == 0 for score in scores) if condition != "reference" else None,
                           "pending_review": sum(score is None for score in scores),
                           "reviewed_episodes": sum(row.get("review") is not None for row in rows)})
        if condition == "reference":
            continue
        rubric_keys = manifest["rubric"]["challenge" if family == "challenge" else "supported"]
        for check in rubric_keys:
            judgments = [row["review"]["checks"][check] for row in rows
                         if row.get("review") is not None and check in row["review"].get("checks", {})]
            diagnostics.append({"split": split, "family": family, "condition": condition,
                                "check": check, "passes": sum(value is True for value in judgments),
                                "reviewed_denominator": len(judgments),
                                "machine_failures": machine_failures,
                                "pending_structural_successes": sum(score is None for score in scores)})
    usage = []
    for condition in ("reference", "generic", "structured"):
        subset = [row for row in records.values() if row["condition"] == condition]
        for field, label in USAGE_FIELDS.items():
            values = []
            for row in subset:
                value = row.get("usage", {}).get(field)
                if field == "active_seconds" and condition == "reference" and value is None:
                    value = row.get("result", {}).get("elapsed_seconds")
                if _number(value):
                    values.append(value)
            usage.append({"condition": condition, "field": field, "label": label,
                          "mean": statistics.mean(values) if values else None,
                          "minimum": min(values) if values else None,
                          "maximum": max(values) if values else None,
                          "measured_episodes": len(values), "unavailable_episodes": len(subset) - len(values),
                          "estimated_token_usage_episodes": sum(bool(row.get("usage", {}).get("token_usage_estimated")) for row in subset)
                          if field == "completion_tokens" else None})
    failures = Counter()
    for row in records.values():
        if row.get("machine", {}).get("pass", False):
            continue
        reason = (row.get("usage", {}).get("stop_reason") or row.get("error") or
                  row.get("result", {}).get("error"))
        if not reason:
            reason = "; ".join(str(item) for item in row.get("machine", {}).get("errors", [])) or "unrecorded_failure_reason"
        failures[(row["split"], row["family"], row["condition"], reason)] += 1
    failure_rows = [{"split": split, "family": family, "condition": condition, "reason": reason, "episodes": count}
                    for (split, family, condition, reason), count in sorted(failures.items())]
    case_means = []
    for case in cases:
        if case["split"] != "heldout":
            continue
        for condition in ("generic", "structured"):
            repeats = manifest["repeats"]["heldout"]
            identifiers = [f"{case['case_id']}_{condition}_{repeat}" for repeat in range(repeats)]
            rows = [records[identifier] for identifier in identifiers if identifier in records]
            scores = [completion(row) for row in rows]
            mean = statistics.mean(scores) if len(scores) == repeats and all(score is not None for score in scores) else None
            case_means.append({"case_id": case["case_id"], "family": case["family"], "condition": condition,
                               "expected_repeats": repeats, "completed_repeats": len(rows),
                               "pending_reviews": sum(score is None for score in scores),
                               "primary_mean": mean, "episode_ids": identifiers})
    reviewers = sorted({(row["review"]["reviewer_type"], row["review"]["reviewer"], row["review"]["rubric_version"])
                        for row in records.values() if row.get("review")})
    summary = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign.name, "model": manifest["backend"]["model"],
        "seed_offset": manifest.get("seed_offset", 0),
        "manifest_sha256": sha256((campaign / "manifest.json").read_bytes()).hexdigest(),
        "record_sha256_at_read": hashes,
        "snapshot_note": "Records are read atomically one at a time. A running campaign may advance after this snapshot.",
        "required_expected": len(required), "required_completed": required_complete,
        "required_running": len(required_running), "required_missing": len(missing),
        "complete_execution": required_complete == len(required),
        "qualitative_pending": sum(completion(row) is None for row in records.values() if row["condition"] != "reference"),
        "missing_episode_ids": missing, "running_episode_ids": required_running,
        "groups": aggregates, "review_diagnostics": diagnostics, "usage": usage,
        "failure_reasons": failure_rows, "heldout_case_primary_means": case_means,
        "review_provenance": [{"type": kind, "reviewer": who, "rubric_version": version} for kind, who, version in reviewers],
        "review_policy": "Identified unblinded AI assessment; not independent human review or a human-user study. Inspect per-episode rationales.",
        "diagnostic_policy": "Only recorded rubric checks are aggregated. Appropriate task/target is one combined check. No unreviewed next-action taxonomy is inferred.",
        "episode_details": "records/<episode_id>.json contains response evidence, next_action, and identified review rationale; artifacts/<episode_id>/sessions/<episode_id>.json contains the recorded tool sequence.",
    }
    lines = ["# Derived campaign summary", "", f"Campaign: `{campaign.name}`. Model: `{summary['model']}`. Seed offset: `{summary['seed_offset']}`.", "",
             f"**Execution {'complete' if summary['complete_execution'] else 'incomplete'}:** {required_complete}/{len(required)} required records completed; {len(required_running)} running; {len(missing)} missing. Completed structural successes awaiting qualitative review: {summary['qualitative_pending']}.", "",
             "This is a read-only derived snapshot, not a new inference run. Running campaigns can advance after records are read. JSON includes manifest and record hashes for the snapshot. Optional challenge-development episodes do not replace required episodes.", "",
             "## Completion", "", "Primary failure is zero; structurally successful but unreviewed is pending, never a pass. References have no conversational primary score. Counts use the frozen episode assignments.", "",
             "| Split | Family | Condition | Expected | Completed | Structural passes | Primary passes | Primary failures | Pending review |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        fields = [row[key] if row[key] is not None else "N/A" for key in ("split", "family", "condition", "expected", "completed", "structural_passes", "primary_passes", "primary_failures", "pending_review")]
        lines.append("| " + " | ".join(_cell(value) for value in fields) + " |")
    lines += ["", "## Reviewed diagnostics", "", summary["diagnostic_policy"],
              "Pass counts below use only episodes where that check was explicitly reviewed. Machine failures and pending structural successes are shown separately; neither is silently counted as a reviewed success.", "",
              "| Split | Family | Condition | Recorded check | Pass / reviewed | Machine failures | Pending structural successes |", "|---|---|---|---|---:|---:|---:|"]
    for row in diagnostics:
        lines.append(f"| {row['split']} | {row['family']} | {row['condition']} | {row['check']} | {row['passes']} / {row['reviewed_denominator']} | {row['machine_failures']} | {row['pending_structural_successes']} |")
    lines += ["", "Detailed next-action assessment remains in `records/<episode_id>.json`: inspect both delivered responses, `next_action`, selected artifact IDs, and the identified review rationale. The session file under `artifacts/<episode_id>/sessions/` records actual tools. No comparison/clarification/refusal taxonomy was retrospectively guessed from wording.", "",
              "## Usage across completed records", "", "Means and ranges pool completed records across recorded splits by condition and are descriptive, not paired quality estimates. Missing counters are unavailable, not zero. Reference latency uses its recorded result duration; absent reference counters remain unavailable. Generated tokens include conservative reservations when usage was unknown; estimated-usage episodes are identified in JSON. Local hardware/electricity cost was not measured.", "",
              "| Condition | Measure | Mean | Minimum | Maximum | Measured | Unavailable |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in usage:
        numeric = [f"{row[key]:.3f}" if row[key] is not None else "unavailable" for key in ("mean", "minimum", "maximum")]
        lines.append(f"| {row['condition']} | {row['label']} | {' | '.join(numeric)} | {row['measured_episodes']} | {row['unavailable_episodes']} |")
    lines += ["", "## Recorded structural or execution failures", "", "Reasons below are the actual recorded final stop/error or structural-check messages. Recovered intermediate faults are not inferred as episode failures. Qualitative failures are represented in the review diagnostics above.", "",
              "| Split | Family | Condition | Recorded reason | Episodes |", "|---|---|---|---|---:|"]
    for row in failure_rows:
        lines.append(f"| {row['split']} | {row['family']} | {row['condition']} | {_cell(row['reason'])} | {row['episodes']} |")
    lines += ["", "## Held-out primary completion by case", "", "The case mean is available only when every frozen repeat is complete and scored. A failed repeat contributes zero. Pending or missing repeats prevent a mean; they are not dropped. Repeats do not create independent datasets.", "",
              "| Case | Family | Condition | Completed / expected repeats | Pending reviews | Primary mean |", "|---|---|---|---:|---:|---:|"]
    for row in case_means:
        mean = f"{row['primary_mean']:.3f}" if row['primary_mean'] is not None else "unavailable"
        lines.append(f"| {row['case_id']} | {row['family']} | {row['condition']} | {row['completed_repeats']} / {row['expected_repeats']} | {row['pending_reviews']} | {mean} |")
    lines += ["", "## Review provenance and limits", "", summary["review_policy"], ""]
    lines += [f"- {_cell(kind)}: {_cell(who)}; rubric {_cell(version)}." for kind, who, version in reviewers] or ["No identified qualitative reviews in this snapshot."]
    lines += ["", "No causal, human-outcome, production-reliability, or general-superiority claim follows from these synthetic cases. Different families' model-quality metrics are not pooled by this script; inspect the campaign's original REPORT.md and per-episode metrics for those results.", ""]
    destination.mkdir(parents=True, exist_ok=True)
    _write(destination / "aggregate.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write(destination / "AGGREGATE.md", "\n".join(lines))
    return {"markdown": str(destination / "AGGREGATE.md"), "json": str(destination / "aggregate.json"),
            "required_expected": len(required), "required_completed": required_complete,
            "required_running": len(required_running), "required_missing": len(missing),
            "qualitative_pending": summary["qualitative_pending"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize_results(args.campaign, args.destination), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Summary failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
