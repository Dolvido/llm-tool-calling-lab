import json
import time

import pytest

from llm_tool_calling_lab.backend import public_text
from llm_tool_calling_lab.chat import Session, COMMON_PROMPT, STRUCTURED_PROMPT
from llm_tool_calling_lab.contracts import LabConfig, RunResult, TaskSpec


class FakeService:
    def __init__(self, root):
        self.root = root
        self.runs = {}
        self.timeouts = []

    def inspect_dataset(self, identifier):
        if identifier != "data_a":
            raise ValueError("Unknown dataset")
        return {"columns": ["x1", "x2", "target"], "rows": 200}

    def list_methods(self, family=None):
        return [{"method_id": "ridge", "family": "regression"},
                {"method_id": "random_forest_regressor", "family": "regression"}]

    def run_candidate(self, task, method_id, seed=0, timeout_s=60):
        self.timeouts.append(timeout_s)
        run = RunResult(run_id=f"run_{len(self.runs) + 1}", task=task,
                        method_id=method_id, status="ok", metrics={"mae": 0.5},
                        summary={"predictions": list(range(10000))},
                        artifact_paths={"model": "/private/model.pkl"})
        self.runs[run.run_id] = run.model_dump(mode="json")
        return run

    def retrieve_result(self, identifier):
        return self.runs[identifier]

    def compare_results(self, identifiers):
        return {"run_ids": identifiers, "preferred_run_id": identifiers[0]}


class FakeBackend:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, tools, *, timeout_s, max_tokens):
        self.requests.append({"messages": json.loads(json.dumps(messages)),
                              "tools": tools, "timeout_s": timeout_s,
                              "max_tokens": max_tokens})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def answer(ids=(), selected=None, content="Here is the recorded evidence.", tokens=20):
    return {"content": json.dumps({"answer": content, "evidence_ids": list(ids),
                                    "selected_run_id": selected, "next_action": "Review limitations."}),
            "completion_tokens": tokens, "prompt_tokens": 100}


def tool(name="run_candidate", args=None, tokens=30):
    if args is None:
        args = {"family": "regression", "dataset_id": "data_a",
                         "features": ["x1", "x2"], "target": "target",
                "method_id": "ridge"}
    return {"content": "I will fit a regression model.", "tool_calls": [
        {"function": {"name": name, "arguments": args}}],
        "completion_tokens": tokens, "prompt_tokens": 100}


def session(tmp_path, *responses, **overrides):
    return Session(FakeService(tmp_path), LabConfig(**overrides),
                   backend=FakeBackend(*responses))


def test_native_tool_round_trip_and_persisted_evidence(tmp_path):
    s = session(tmp_path, tool(), answer(["run_1"], "run_1"))
    result = s.respond("Predict target", "data_a")
    assert result["status"] == "ok"
    assert result["evidence"][0]["metrics"]["mae"] == .5
    assert s.state["fits"] == 1 and s.state["llm_responses"] == 2
    stored = json.loads(s.path.read_text(encoding="utf-8"))
    assert stored["last_response"]["selected_run_id"] == "run_1"
    tool_message = s.backend.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "predictions" not in tool_message["content"]
    assert "artifact_paths" not in tool_message["content"]


def test_follow_up_shares_fit_and_response_budgets(tmp_path):
    s = session(tmp_path, tool(), answer(["run_1"], "run_1"),
                tool(), answer(["run_1", "run_2"], "run_2"),
                tool(), answer(["run_2"], "run_2"))
    assert s.respond("Predict target", "data_a")["status"] == "ok"
    assert s.respond("Compare another method")["status"] == "ok"
    result = s.respond("Fit again")
    assert s.state["fits"] == 2
    assert len(s.service.runs) == 2
    assert s.state["repairs"] == 1
    assert result["status"] == "ok"  # The model can acknowledge existing evidence.
    assert s.respond("Another question")["status"] == "error"
    assert s.state["llm_responses"] == 6


def test_unknown_tools_are_counted_and_never_executed(tmp_path):
    s = session(tmp_path, tool("execute_code", {"code": "print('no')"}),
                tool("execute_code", {"code": "print('no')"}))
    result = s.respond("Compute", "data_a")
    assert result["status"] == "error"
    assert s.state["tool_calls"] == 2 and s.state["repairs"] == 1
    assert s.state["fits"] == 0


def test_cross_session_results_and_datasets_rejected(tmp_path):
    s = session(tmp_path, tool("retrieve_result", {"run_id": "run_foreign"}),
                tool("inspect_dataset", {"dataset_id": "../../private.csv"}))
    assert s.respond("Inspect", "data_a")["status"] == "error"
    assert s.state["fits"] == 0
    assert all("not registered" in event["output"]["error"]
               for event in s.events if event["kind"] == "tool_result")


def test_extra_tool_arguments_rejected(tmp_path):
    s = session(tmp_path, tool("inspect_dataset", {"dataset_id": "data_a", "path": "secret.csv"}),
                answer(content="Please check your input."))
    assert s.respond("Inspect", "data_a")["status"] == "ok"
    assert s.state["repairs"] == 1 and s.state["tool_calls"] == 1


def test_fabricated_final_citation_cannot_be_accepted(tmp_path):
    s = session(tmp_path, answer(["run_invented"], "run_invented"),
                answer(["run_invented"], "run_invented"))
    assert s.respond("Predict", "data_a")["status"] == "error"
    assert s.state["repairs"] == 1
    assert not s.results


def test_tool_budget_counts_invalid_attempts(tmp_path):
    s = session(tmp_path, tool("bad", {}), tool(), max_tool_calls=1)
    assert s.respond("Predict", "data_a")["status"] == "error"
    assert s.state["tool_calls"] == 1 and s.state["fits"] == 0


def test_generated_tokens_cap_is_cumulative_and_reserves_remainder(tmp_path):
    s = session(tmp_path, answer(tokens=20), answer(tokens=5), max_episode_tokens=25)
    assert s.respond("Clarify", "data_a")["status"] == "ok"
    assert s.respond("Thanks")["status"] == "ok"
    assert s.backend.requests[1]["max_tokens"] == 5
    assert s.respond("More")["error"] == "generated-token budget exhausted"


def test_backend_failure_keeps_unknown_token_reservation(tmp_path):
    s = session(tmp_path, ConnectionError("connection refused"),
                answer(tokens=5), max_output_tokens=100, max_episode_tokens=105)
    assert s.respond("Clarify", "data_a")["status"] == "ok"
    assert s.state["completion_tokens"] == 105
    assert s.state["token_usage_estimated"] is True
    assert s.backend.requests[1]["max_tokens"] == 5


def test_unknown_usage_is_conservatively_reserved(tmp_path):
    response = answer()
    response.pop("completion_tokens")
    s = session(tmp_path, response, max_output_tokens=50, max_episode_tokens=50)
    s.respond("Clarify", "data_a")
    assert s.state["completion_tokens"] == 50
    assert s.respond("More")["status"] == "error"


def test_human_pause_does_not_consume_execution_budget(tmp_path, monkeypatch):
    clock = [1.0]
    monkeypatch.setattr("llm_tool_calling_lab.chat.time.monotonic", lambda: clock[0])
    s = session(tmp_path, answer(), answer(), max_seconds=1)
    assert s.respond("Clarify", "data_a")["status"] == "ok"
    clock[0] = 10000.0
    assert s.respond("Clarify again")["status"] == "ok"
    assert s.state["active_seconds"] == 0


def test_late_backend_response_cannot_run_tools(tmp_path):
    s = session(tmp_path, max_seconds=0.01)
    class Slow:
        def chat(self, *args, **kwargs):
            time.sleep(.02)
            return tool()
    s.backend = Slow()
    assert s.respond("Predict", "data_a")["status"] == "error"
    assert s.state["fits"] == 0


def test_fit_receives_remaining_deadline(tmp_path):
    s = session(tmp_path, tool(), answer(["run_1"], "run_1"), max_seconds=20)
    s.respond("Predict", "data_a")
    assert 0 < s.service.timeouts[0] <= 20


def test_no_private_reasoning_logged(tmp_path):
    s = session(tmp_path, dict(answer(), content="<think>PRIVATE REASONING</think>" + answer()["content"]))
    assert s.respond("Clarify", "data_a")["status"] == "ok"
    assert "PRIVATE REASONING" not in s.path.read_text(encoding="utf-8")
    assert public_text("Visible <think>unclosed private") == "Visible"


def test_conditions_share_tools_and_base_context(tmp_path):
    a = Session(FakeService(tmp_path), LabConfig(), "structured", FakeBackend(answer()))
    b = Session(FakeService(tmp_path), LabConfig(), "generic", FakeBackend(answer()))
    a.respond("Clarify", "data_a")
    b.respond("Clarify", "data_a")
    assert a.backend.requests[0]["tools"] == b.backend.requests[0]["tools"]
    assert a.messages[1]["content"] == b.messages[1]["content"]
    assert a.messages[0]["content"] == COMMON_PROMPT + STRUCTURED_PROMPT
    assert b.messages[0]["content"] == COMMON_PROMPT


def test_preexisting_session_is_not_overwritten(tmp_path):
    Session(FakeService(tmp_path), LabConfig(), session_id="retained_run")
    with pytest.raises(ValueError, match="already exists"):
        Session(FakeService(tmp_path), LabConfig(), session_id="retained_run")


def test_invalid_session_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid session"):
        Session(FakeService(tmp_path), LabConfig(), session_id="../escape")


def test_failed_fit_consumes_fit_reservation_and_is_retained(tmp_path):
    s = session(tmp_path, tool(), answer(content="The fit failed; inspect the data."))
    def fail(task, method_id, **kwargs):
        return RunResult(run_id="run_failed", task=task, method_id=method_id,
                         status="error", error="invalid class support")
    s.service.run_candidate = fail
    assert s.respond("Predict", "data_a")["status"] == "ok"
    assert s.state["fits"] == 1 and s.results[0]["status"] == "error"


def test_response_counter_cannot_reset_on_followup(tmp_path):
    s = session(tmp_path, answer(), max_llm_responses=1)
    assert s.respond("Clarify", "data_a")["status"] == "ok"
    assert s.respond("Clarify again")["error"] == "LLM-response budget exhausted"
    assert len(s.backend.requests) == 1


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_anomaly_answer_requires_actual_top_ids_even_for_small_batch(tmp_path, batch_size):
    top = [str(i) for i in range(min(4, batch_size))]
    final = answer(["run_1"], "run_1")
    payload = json.loads(final["content"])
    payload["row_ids"] = top
    final["content"] = json.dumps(payload)
    s = session(tmp_path, final)
    s.results = [{"run_id": "run_1", "status": "ok", "task": {"family": "anomaly"},
                  "summary": {"scoring_rows": batch_size, "ranking": [{"row_id": value} for value in top]}}]
    response = s.respond("Explain the ranking", "data_a")
    assert response["status"] == "ok" and response["row_ids"] == top


def test_fabricated_or_duplicate_anomaly_answer_ids_fail(tmp_path):
    final = answer(["run_1"], "run_1")
    payload = json.loads(final["content"])
    payload["row_ids"] = ["999", "999", "999", "999"]
    final["content"] = json.dumps(payload)
    s = session(tmp_path, final, final)
    s.results = [{"run_id": "run_1", "status": "ok", "task": {"family": "anomaly"},
                  "summary": {"scoring_rows": 64, "ranking": [{"row_id": str(i)} for i in range(4)]}}]
    assert s.respond("Explain the ranking", "data_a")["status"] == "error"
    assert s.state["repairs"] == 1
