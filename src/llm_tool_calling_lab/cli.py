"""Small public command interface. Model inference is explicit, never an install hook."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

from .contracts import LabConfig, TaskSpec

def config_from(path=None):
    return LabConfig.model_validate_json(Path(path).read_text(encoding="utf-8-sig")) if path else LabConfig()

def main():
    parser = argparse.ArgumentParser(prog="lab", description="Local, experimental conversational ML tools")
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check runtime and installed model without inference")
    chat = sub.add_parser("chat", help="Launch the local chat interface")
    chat.add_argument("--port", type=int, default=8501)
    fixtures = sub.add_parser("examples", help="Generate synthetic examples and evaluator fixtures")
    fixtures.add_argument("destination", type=Path, nargs="?", default=Path(".lab/examples"))
    freeze = sub.add_parser("freeze", help="Freeze a new campaign before evaluation")
    freeze.add_argument("destination", type=Path)
    freeze.add_argument("--families", nargs="+", default=["regression", "classification", "anomaly", "clustering"])
    freeze.add_argument("--seed-offset", type=int, default=0, help="Freeze a fresh synthetic dataset assignment")
    evaluate = sub.add_parser("evaluate", help="Run/resume a frozen local-model campaign")
    evaluate.add_argument("destination", type=Path)
    evaluate.add_argument("--split", choices=["development", "heldout", "challenge", "challenge_development"], default="development")
    evaluate.add_argument("--limit", type=int)
    report = sub.add_parser("report", help="Build a report preserving failures and pending reviews")
    report.add_argument("destination", type=Path)
    tutorial = sub.add_parser("tutorial", help="Exercise the ElasticNet replacement on a development example")
    tutorial.add_argument("--root", type=Path, default=Path(".lab/tutorial"))
    args = parser.parse_args()
    config = config_from(args.config)
    try:
        if args.command == "doctor":
            from .evaluation import backend_identity
            from importlib.metadata import version
            info = {"python": sys.version, "backend": backend_identity(config),
                    "packages": {n: version(n) for n in ("scikit-learn", "streamlit", "ollama", "pydantic")},
                    "status": "ready_for_live_smoke", "live_verified": False}
            print(json.dumps(info, indent=2))
        elif args.command == "chat":
            import os
            env = dict(os.environ, LAB_CONFIG_JSON=config.model_dump_json())
            command = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("ui.py")),
                       "--server.address", "127.0.0.1", "--server.port", str(args.port),
                       "--server.headless", "true", "--browser.gatherUsageStats", "false"]
            return subprocess.call(command, env=env)
        elif args.command == "examples":
            from .fixtures import generate_examples
            cases = generate_examples(args.destination)
            print(f"Generated {len(cases)} cases at {args.destination.resolve()}; truth files are evaluator-only.")
        elif args.command == "freeze":
            from .evaluation import freeze
            print(freeze(args.destination, config, args.families, seed_offset=args.seed_offset))
        elif args.command == "evaluate":
            from .evaluation import run_campaign
            run_campaign(args.destination, args.split, args.limit)
        elif args.command == "report":
            from .evaluation import write_report
            print(write_report(args.destination))
        elif args.command == "tutorial":
            from .fixtures import generate_examples
            from .models import ModelService
            generate_examples(args.root / "fixtures", ("regression",))
            records = []
            for catalog, method in (("default", "ridge"), ("tutorial", "elastic_net")):
                service = ModelService(args.root / catalog, catalog=catalog)
                identifier = service.register_csv(args.root / "fixtures" / "c000" / "data.csv")
                task = TaskSpec(family="regression", dataset_id=identifier, features=[f"x{i}" for i in range(1,7)], target="y")
                result = service.run_candidate(task, method)
                records.append(result.model_dump())
                if result.status != "ok":
                    raise RuntimeError(result.error)
            (args.root / "comparison.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(json.dumps([{"method":r["method_id"], "metrics":r["metrics"], "run_id":r["run_id"]} for r in records], indent=2))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
