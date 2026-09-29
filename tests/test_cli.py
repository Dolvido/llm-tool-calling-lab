import json
import sys

import pytest

from llm_tool_calling_lab import cli, evaluation


def invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["lab", *arguments])
    return cli.main()


def test_help_lists_public_commands_without_model_calls(monkeypatch, capsys):
    with pytest.raises(SystemExit) as result:
        invoke(monkeypatch, "--help")
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("doctor", "chat", "examples", "freeze", "evaluate", "report", "tutorial"):
        assert command in help_text


def test_doctor_reports_runtime_without_claiming_live_verification(monkeypatch, capsys):
    checks = []
    def backend_identity(config):
        checks.append(config.model)
        return {"model": config.model, "digest": "test-model-digest", "size": 1}
    monkeypatch.setattr(evaluation, "backend_identity", backend_identity)
    assert invoke(monkeypatch, "doctor") == 0
    output = json.loads(capsys.readouterr().out)
    assert checks == ["qwen3:14b-q4_K_M"]
    assert output["status"] == "ready_for_live_smoke"
    assert output["live_verified"] is False
    assert {"scikit-learn", "streamlit", "ollama", "pydantic"} <= set(output["packages"])


def test_doctor_missing_backend_is_actionable_error(monkeypatch, capsys):
    def missing(config):
        raise ValueError("Configured model is not installed")
    monkeypatch.setattr(evaluation, "backend_identity", missing)
    assert invoke(monkeypatch, "doctor") == 1
    assert "Configured model is not installed" in capsys.readouterr().err


def test_chat_launch_is_local_and_preserves_configuration(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "test-local-model", "max_fits": 1}), encoding="utf-8")
    launches = []
    def launch(command, env):
        launches.append((command, env))
        return 0
    monkeypatch.setattr(cli.subprocess, "call", launch)
    assert invoke(monkeypatch, "--config", str(config), "chat", "--port", "8510") == 0
    command, environment = launches[0]
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--server.port") + 1] == "8510"
    assert command[command.index("--browser.gatherUsageStats") + 1] == "false"
    assert json.loads(environment["LAB_CONFIG_JSON"])["max_fits"] == 1
    assert json.loads(environment["LAB_CONFIG_JSON"])["model"] == "test-local-model"
