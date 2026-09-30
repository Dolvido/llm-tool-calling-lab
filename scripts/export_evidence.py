"""Export a bounded synthetic campaign without model binaries or machine paths."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any
import zipfile


CASE_ID = re.compile(r"c\d{3}\Z")
RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")


def _json(raw: bytes) -> Any:
    def invalid_constant(value):
        raise ValueError("Evidence JSON contains a nonfinite number")
    return json.loads(raw.decode("utf-8-sig"), parse_constant=invalid_constant)


def _inside(root: Path, relative: str) -> Path:
    logical = PurePosixPath(relative)
    if logical.is_absolute() or ".." in logical.parts or "\\" in relative:
        raise ValueError("Unsafe relative evidence path")
    path = root.joinpath(*logical.parts)
    # Reject links and junctions rather than following them into an unrelated file.
    for ancestor in (path, *path.parents):
        if ancestor == root.parent:
            break
        if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
            raise ValueError("Linked evidence paths are not exportable")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Evidence path escapes campaign")
    return path


class Sanitizer:
    def __init__(self, campaign: Path, project_root: Path | None):
        roots = [(campaign, "")]
        if project_root is not None and project_root != campaign:
            roots.append((project_root, "source/"))
        self.replacements = []
        for root, replacement in roots:
            variants = {str(root), root.as_posix(), str(root).replace("\\", "\\\\")}
            for variant in sorted(variants, key=len, reverse=True):
                self.replacements.append((variant.rstrip("/\\"), replacement))

    def text(self, value: str) -> str:
        for prefix, replacement in self.replacements:
            value = re.sub(re.escape(prefix) + r"[\\/]+", lambda match: replacement, value, flags=re.I)
            value = re.sub(re.escape(prefix) + r"(?=$|[\"'\s<>])", lambda match: replacement.rstrip("/") or ".", value, flags=re.I)
        # Normalize only the relative path prefixes introduced or used by this exporter.
        value = re.sub(r"(?:artifacts|fixtures|records|source)[\\/][A-Za-z0-9_.\\/\-]+",
                       lambda match: re.sub(r"[\\/]+", "/", match.group()), value)
        if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:Users|home|mnt|tmp|private)/", value):
            raise ValueError("Unrecognized absolute local path remains in evidence text")
        return value

    def object(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.object(item) for item in value]
        if isinstance(value, dict):
            return {self.text(key): self.object(item) for key, item in value.items()}
        return value


def export_evidence(campaign: Path, output_zip: Path, project_root: Path | None = None) -> dict:
    """Create a release ZIP and checksum; originals and frozen hashes stay untouched.

    Only canonical synthetic fixtures named in the frozen manifest, known episode
    records, their sessions, and registered run JSON are eligible. Unknown files
    are ignored. Running episodes, fixture changes, external paths, and links fail
    closed. This function does not fit models, run inference, or revise a campaign.
    """
    campaign = Path(campaign).resolve(strict=True)
    output_zip = Path(output_zip).resolve()
    if output_zip.is_relative_to(campaign):
        raise ValueError("Write the export outside the frozen campaign directory")
    if output_zip.exists() or Path(str(output_zip) + ".sha256").exists():
        raise ValueError("Export destination already exists")
    if output_zip.suffix.lower() != ".zip":
        raise ValueError("Export destination must end in .zip")
    if project_root is None:
        project_root = campaign.parent.parent if campaign.parent.name == ".lab" else Path(__file__).resolve().parents[1]
    project_root = Path(project_root).resolve()
    sanitizer = Sanitizer(campaign, project_root)
    selected: dict[str, bytes] = {}

    def read(relative: str, required: bool = True) -> bytes | None:
        path = _inside(campaign, relative)
        if not path.is_file():
            if required:
                raise ValueError(f"Required evidence file is missing: {relative}")
            return None
        raw = path.read_bytes()
        selected[relative] = raw
        return raw

    manifest = _json(read("manifest.json"))
    cases_raw = read("fixtures/cases.json")
    expected_cases_digest = manifest.get("fixture_hashes", {}).get("fixtures/cases.json")
    if sha256(cases_raw).hexdigest() != expected_cases_digest:
        raise ValueError("Frozen case manifest is missing or changed")
    cases = _json(cases_raw)
    if not isinstance(cases, list):
        raise ValueError("Synthetic case manifest must be a list")
    fixture_names = {"fixtures/cases.json"}
    episodes = {}
    seen_cases = set()
    for case in cases:
        identifier = case.get("case_id", "")
        if not CASE_ID.fullmatch(identifier) or identifier in seen_cases:
            raise ValueError("Invalid or duplicate synthetic case ID")
        seen_cases.add(identifier)
        if case.get("data") != f"{identifier}/data.csv" or case.get("scoring_data") not in {None, f"{identifier}/batch.csv"}:
            raise ValueError("Noncanonical synthetic fixture reference")
        fixture_names.add("fixtures/" + case["data"])
        if case.get("scoring_data"):
            fixture_names.add("fixtures/" + case["scoring_data"])
        split = case.get("split")
        repeats = manifest.get("repeats", {}).get(split)
        if type(repeats) is not int or not 1 <= repeats <= 2:
            raise ValueError("Unsupported frozen repetition count")
        for condition in ("reference", "generic", "structured"):
            if condition == "reference" and case.get("family") == "challenge":
                continue
            for repeat in range(1 if condition == "reference" else repeats):
                episodes[f"{identifier}_{condition}_{repeat}"] = (case, condition)
    for relative in sorted(fixture_names):
        raw = read(relative)
        if sha256(raw).hexdigest() != manifest.get("fixture_hashes", {}).get(relative):
            raise ValueError(f"Frozen fixture is missing or changed: {relative}")

    reviewers = set()
    counts = {"records": 0, "failed_records": 0, "missing_records": 0, "sessions": 0, "runs": 0,
              "required_expected": sum(case["split"] != "challenge_development" for case, _ in episodes.values()),
              "required_records": 0, "required_missing": 0}
    for episode_id, (case, condition) in sorted(episodes.items()):
        raw = read(f"records/{episode_id}.json", required=False)
        if raw is None:
            counts["missing_records"] += 1
            counts["required_missing"] += case["split"] != "challenge_development"
            continue
        record = _json(raw)
        if record.get("episode_id") != episode_id or record.get("case_id") != case["case_id"] or record.get("condition") != condition:
            raise ValueError(f"Record identity does not match its fixture: {episode_id}")
        if record.get("status") not in {"ok", "error"}:
            raise ValueError(f"Episode has not completed or been recovered: {episode_id}")
        counts["records"] += 1
        counts["required_records"] += case["split"] != "challenge_development"
        counts["failed_records"] += record["status"] == "error"
        for review in record.get("review_history", []) + ([record["review"]] if record.get("review") else []):
            if review.get("reviewer_type") not in {"AI", "human"}:
                raise ValueError("Unknown qualitative review provenance")
            reviewers.add((review["reviewer_type"], review.get("reviewer", "unspecified")))
        registered_runs = []
        if condition == "reference" and record.get("result"):
            registered_runs.append(record["result"].get("run_id"))
        if condition != "reference":
            canonical = f"artifacts/{episode_id}/sessions/{episode_id}.json"
            declared = record.get("session_file")
            if declared is not None and declared != canonical:
                raise ValueError(f"Noncanonical session reference: {episode_id}")
            session_raw = read(canonical, required=declared is not None)
            if session_raw is not None:
                session = _json(session_raw)
                if session.get("session_id") != episode_id:
                    raise ValueError(f"Session identity mismatch: {episode_id}")
                registered_runs.extend(result.get("run_id") for result in session.get("results", []))
                counts["sessions"] += 1
        for run_id in sorted(set(registered_runs)):
            if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
                raise ValueError(f"Invalid registered run in episode: {episode_id}")
            directory = f"artifacts/{episode_id}/runs/{run_id}"
            result = _json(read(directory + "/result.json"))
            if result.get("run_id") != run_id:
                raise ValueError(f"Run identity mismatch: {episode_id}")
            for filename in ("predictions.json", "private.json"):
                read(directory + "/" + filename, required=False)
            counts["runs"] += 1
    for relative in ("REPORT.md", "summary.json", "CAMPAIGN_PROVENANCE.json", "STARTUP_FAILURE.json"):
        read(relative, required=False)

    exported = {}
    index_entries = []
    for relative, original in sorted(selected.items()):
        if relative.endswith(".json"):
            parsed = _json(original)
            sanitized = sanitizer.object(parsed)
            payload = original if sanitized == parsed else (json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            category = "json"
        elif relative.endswith(".md"):
            payload = sanitizer.text(original.decode("utf-8")).encode("utf-8")
            category = "markdown"
        else:
            payload, category = original, "csv"
        exported[relative] = payload
        index_entries.append({"path": relative, "kind": category,
                              "original_sha256": sha256(original).hexdigest(),
                              "exported_sha256": sha256(payload).hexdigest(),
                              "original_bytes": len(original), "exported_bytes": len(payload),
                              "sanitized": payload != original})
    review_provenance = [{"type": kind, "reviewer": sanitizer.text(name)} for kind, name in sorted(reviewers)]
    index = {"schema_version": 1, "synthetic_campaign": True, "counts": counts,
             "review_provenance": review_provenance,
             "fixture_csv_bytes_preserved": True,
             "excluded": ["fitted model binaries", "registered dataset stores", "locks", "temporary files", "unknown files"],
             "files": index_entries}
    exported["export_index.json"] = (json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    reviewer_types = ", ".join(sorted({entry["type"] for entry in review_provenance})) or "none recorded"
    exported["README.md"] = f"""# Synthetic campaign evidence

This archive contains the frozen generated numerical fixtures and recorded execution evidence for one campaign. It contains no uploaded user tables or fitted model binaries. Case definitions in `fixtures/cases.json` include **evaluator-only ground truth**; `private.json` contains evaluator-only partitions, preprocessing information, and test outputs. These fields were not supplied to the chatbot. Public tool outputs and the session transcripts show the evidence it could access.

The original campaign manifest preserves source fingerprints, package versions, model identity, fixture hashes, budgets, and case assignments. Failed records and the failure history in exported sessions are retained; the exporter does not rerun or repair episodes. Earlier development campaigns remain separate archives. This export includes {counts['records']} records, including {counts['failed_records']} errors. Required campaign records: {counts['required_records']}/{counts['required_expected']}, with {counts['required_missing']} required records missing. There are {counts['missing_records']} absent possible fixture episodes in total, including any optional challenge-development episodes not run. Consult the campaign report for completion and qualitative review status.

Absolute campaign paths have been converted to references relative to this archive. References to other project files use the `source/` prefix and do not imply those source files are included. Identifiers, metric values, and numerical claims are unchanged. `export_index.json` records every included original file's SHA256 and exported SHA256. A changed JSON or Markdown hash reflects path sanitization; the original manifest continues to describe the originals. CSV bytes and their frozen hashes are preserved exactly. The export index and this README are new export metadata rather than original campaign files.

Recorded qualitative reviewer types: **{reviewer_types}**. Each record retains reviewer identity, type, rationale, and review history when present. AI review is not human review, and missing review is not a qualitative pass. This archive makes no independent claim of model superiority or general reliability.

Files are selected by an allowlist. Optional campaign-provenance and startup-failure receipts are included when present. Model binaries, dataset stores, lock files, temporary files, unrelated notes, and unknown files are excluded. The accompanying `.sha256` file verifies the complete ZIP. Inspect the source repository at the frozen implementation revision to reproduce the campaign. Any source archive is a separate release asset, not nested inside this evidence ZIP.
""".encode("utf-8")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=output_zip.name + ".", suffix=".tmp", dir=output_zip.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, payload in sorted(exported.items()):
                archive.writestr(relative, payload)
        digest = sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(output_zip)
        checksum = Path(str(output_zip) + ".sha256")
        checksum.write_text(f"{digest}  {output_zip.name}\n", encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
    return {"zip": str(output_zip), "sha256_file": str(checksum), "sha256": digest, "files": len(exported), **counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument("--project-root", type=Path)
    arguments = parser.parse_args(argv)
    try:
        print(json.dumps(export_evidence(arguments.campaign, arguments.output_zip, arguments.project_root), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
