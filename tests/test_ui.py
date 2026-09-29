import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import llm_tool_calling_lab.chat as chat


UI_FILE = Path(chat.__file__).with_name("ui.py")


class ClarifyingBackend:
    """Exercise real Session persistence without a model or inference service."""
    def __init__(self, config):
        self.config = config
        self.calls = 0

    def chat(self, messages, tools, *, timeout_s, max_tokens):
        self.calls += 1
        return {"content": json.dumps({"answer": "Which outcome would you like to predict?", "evidence_ids": [], "selected_run_id": None,
                                        "next_action": "Identify the target column before fitting."}),
                "completion_tokens": 30, "prompt_tokens": 100}


class FittingBackend(ClarifyingBackend):
    def chat(self, messages, tools, *, timeout_s, max_tokens):
        if messages[-1]["role"] == "tool":
            evidence = json.loads(messages[-1]["content"])
            return {"content": json.dumps({"answer": "The fitted regression result is shown in the evidence panel; validation does not establish future performance.",
                    "evidence_ids": [evidence["run_id"]], "selected_run_id": evidence["run_id"], "next_action": "Collect representative data before using the predictions."}),
                    "completion_tokens": 40, "prompt_tokens": 100}
        context = json.loads(messages[-1]["content"].split("Registered context (data, not instructions):\n")[-1])
        return {"content": "I will fit a regression model.", "tool_calls": [{"function": {"name": "run_candidate", "arguments": {
            "method_id": "ridge", "family": "regression", "dataset_id": context["dataset"]["id"], "features": [f"x{i}" for i in range(1, 7)], "target": "y"}}}],
            "completion_tokens": 40, "prompt_tokens": 100}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LAB_CONFIG_JSON", raising=False)
    monkeypatch.setattr(chat, "OllamaBackend", ClarifyingBackend)
    return AppTest.from_file(str(UI_FILE), default_timeout=15)


def test_initial_screen_describes_scope_and_does_not_start_inference(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "LLM Tool Calling Lab"
    assert "numerical tables" in app.info[0].value
    assert app.selectbox[0].options == ["Predict a number", "Predict a category", "Find unusual rows", "Explore groups"]
    assert len(app.chat_input) == 0
    assert "lab_session" not in app.session_state


def test_start_followup_and_explicit_reset_share_and_then_reset_episode(app):
    app.run()
    app.button[0].click().run()
    assert not app.exception
    first = app.session_state["lab_session"]
    assert first.backend.calls == 0
    assert first.state["llm_responses"] == 0
    assert len(app.chat_input) == 1
    next(button for button in app.button if button.label == "Try the example question").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert app.session_state["lab_session"].state["llm_responses"] == 1
    app.chat_input[0].set_value("Use the outcome column.").run()
    assert not app.exception
    assert len(app.chat_message) == 4
    assert app.session_state["lab_session"].session_id == first.session_id
    assert app.session_state["lab_session"].state["llm_responses"] == 2
    saved = json.loads(first.path.read_text(encoding="utf-8"))
    assert saved["state"]["llm_responses"] == 2
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["lab_session"].session_id != first.session_id
    assert app.session_state["lab_history"] == []
    assert first.path.exists(), "Starting a new episode must preserve previous evidence."


def test_upload_mode_requires_a_file_before_start(app):
    app.run()
    app.radio[0].set_value("Upload CSV").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Choose a training/reference CSV first."
    assert "lab_session" not in app.session_state


def test_displayed_limits_match_configuration(app, monkeypatch):
    monkeypatch.setenv("LAB_CONFIG_JSON", json.dumps({"max_fits": 1, "max_tool_calls": 3, "max_llm_responses": 2, "max_seconds": 30}))
    app.run()
    assert not app.exception
    assert any("1 fits · 3 tool calls · 2 model responses · 30 seconds" in item.value for item in app.markdown)
    app.button[0].click().run()
    assert not app.exception
    assert any("0/1 fits · 0/3 tools · 0/2 model responses" in item.value for item in app.caption)


def test_evidence_panel_renders_saved_model_metrics(app, monkeypatch):
    monkeypatch.setattr(chat, "OllamaBackend", FittingBackend)
    app.run()
    app.button[0].click().run()
    next(button for button in app.button if button.label == "Try the example question").click().run(timeout=30)
    assert not app.exception
    session = app.session_state["lab_session"]
    assert session.state["fits"] == 1
    result = session.results[0]
    assert result["status"] == "ok", result.get("error")
    rendered = app.dataframe[0].value.set_index("Measure")["Value"].to_dict()
    assert rendered == result["metrics"]
    assert any(result["run_id"] in caption.value for caption in app.caption)
