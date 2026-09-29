"""Local Streamlit interface. Every evidence value comes from a recorded artifact."""
import json
import os
from pathlib import Path
import uuid
import streamlit as st

from llm_tool_calling_lab.contracts import LabConfig
from llm_tool_calling_lab.chat import Session
from llm_tool_calling_lab.models import ModelService
from llm_tool_calling_lab.fixtures import generate_examples

st.set_page_config(page_title="LLM Tool Calling Lab", page_icon="🔬", layout="wide")
st.title("LLM Tool Calling Lab")
st.caption("Ask a question. Compare specialist models. Inspect the evidence.")
st.info("Experimental · numerical tables · local model. Results support exploration, not causal claims or guarantees.")
root = Path(".lab/app")
root.mkdir(parents=True, exist_ok=True)
config = LabConfig.model_validate_json(os.environ.get("LAB_CONFIG_JSON", "{}"))
if not (root / "examples" / "cases.json").exists():
    generate_examples(root / "examples")
cases = json.loads((root / "examples" / "cases.json").read_text(encoding="utf-8"))
examples = {"Predict a number": "c000", "Predict a category": "c010", "Find unusual rows": "c020", "Explore groups": "c030"}
with st.sidebar:
    st.header("Your data")
    mode = st.radio("Choose a starting point", ["Included example", "Upload CSV"])
    case = None
    upload = batch = None
    if mode == "Included example":
        example = st.selectbox("Example", list(examples))
        case = next(c for c in cases if c["case_id"] == examples[example])
        st.caption("Generated demonstration data. No private data is required.")
    else:
        upload = st.file_uploader("Training or reference table", type="csv")
        batch = st.file_uploader("New rows to score (for unusual-row detection)", type="csv")
        st.caption("Up to 10 MB; 100–10,000 training/reference rows; 2–30 numerical features. Tell the chat which column is the target.")
    start = st.button("Start new episode", type="primary")
    st.caption(f"An episode has {config.max_fits} model fits shared across follow-ups. Starting a new episode is explicit.")
    with st.expander("Model and limits"):
        st.write(config.model)
        st.write(f"{config.max_fits} fits · {config.max_tool_calls} tool calls · {config.max_llm_responses} model responses · {config.max_seconds:g} seconds of system work")

if start:
    try:
        service = ModelService(root / "work", catalog=config.catalog)
        if case:
            data_id = service.register_csv(root / "examples" / case["data"])
            score_id = service.register_csv(root / "examples" / case["scoring_data"]) if case["scoring_data"] else None
        else:
            if upload is None:
                raise ValueError("Choose a training/reference CSV first.")
            identifiers = []
            for uploaded in (upload, batch):
                if uploaded is None:
                    identifiers.append(None)
                    continue
                if uploaded.size > 10 * 1024 * 1024:
                    raise ValueError("CSV files must be at most 10 MB.")
                path = root / ("upload_" + uuid.uuid4().hex + ".csv")
                path.write_bytes(uploaded.getvalue())
                try:
                    identifiers.append(service.register_csv(path))
                finally:
                    path.unlink(missing_ok=True)
            data_id, score_id = identifiers
        st.session_state.lab_session = Session(service, config)
        st.session_state.lab_ids = (data_id, score_id)
        st.session_state.lab_history = []
        st.session_state.example_question = case["question"] if case else None
        st.session_state.bound_example = example if case else "Your uploaded data"
    except Exception as exc:
        st.error(str(exc))

if "lab_session" not in st.session_state:
    st.write("Choose an example or upload a table, then select **Start new episode**.")
else:
    session = st.session_state.lab_session
    st.caption("Current episode: " + st.session_state.bound_example)
    for message in st.session_state.lab_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("row_ids"):
                st.write("Selected row IDs: " + ", ".join(message["row_ids"]))
            if message.get("next_action"):
                st.caption("Next step: " + message["next_action"])
            for evidence in message.get("evidence", []):
                with st.expander("Evidence · " + evidence["method_id"], expanded=True):
                    if evidence["metrics"]:
                        st.dataframe([{"Measure":key, "Value":value} for key,value in evidence["metrics"].items()], hide_index=True)
                    st.json(evidence["summary"], expanded=False)
                    for warning in evidence["warnings"]:
                        st.caption(warning)
                    st.caption("Recorded result: " + evidence["run_id"])
    prompt = st.chat_input("Ask about this data or follow up on the evidence")
    if not st.session_state.lab_history and st.session_state.example_question:
        if st.button("Try the example question"):
            prompt = st.session_state.example_question
    if prompt:
        st.session_state.lab_history.append({"role": "user", "content": prompt})
        with st.spinner("Checking the question and working with the available evidence…"):
            response = session.respond(prompt, *st.session_state.lab_ids)
        st.session_state.lab_history.append({"role": "assistant", "content": response["answer"],
            "next_action": response.get("next_action", ""), "evidence": response.get("evidence", []),
            "row_ids": response.get("row_ids", [])})
        st.rerun()
    st.caption(f"Episode usage: {session.state['fits']}/{config.max_fits} fits · {session.state['tool_calls']}/{config.max_tool_calls} tools · {session.state['llm_responses']}/{config.max_llm_responses} model responses")
    st.download_button("Download this episode", session.path.read_text(encoding="utf-8"), file_name=session.path.name, mime="application/json")
