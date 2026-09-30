# LLM Tool Calling Lab Prototype Plan

Prepared September 29, 2026. Implementation and the full frozen evaluation are complete. This document preserves the design rationale; README.md describes the implemented interface and RELEASE_STATUS.md records verified delivery.

## Accepted v0.1 release decisions

Target a public experimental GitHub v0.1.0 release by October 18, within 39 active development hours. Target all four families; at least two verified families are required. Declare any scope reductions before held-out evaluation. Primary backend is local Ollama qwen3:14b-q4_K_M, with Python 3.12 and Streamlit. Defaults are two fits, eight tool calls, six LLM responses, one repair, 180 seconds cumulative system time excluding human pauses, 8,192 context tokens, and 4,096 generated tokens per episode with 1,024 per response. The tutorial replaces Ridge with ElasticNet outside the default benchmark. Earlier tentative implementation choices below are superseded by these decisions and the checked source.

## Purpose and confirmed decisions

Build an open-source, customizable, working prototype that lets people investigate how a conversational LLM can choose and use specialist ML models. Its value is a working reference, replaceable parts, reproducible evidence, and clear limits. It should reduce uncertainty about building and extending this approach. It does not promise to remove uncertainty about future events.

The user confirmed breadth across as many generalizable approaches as practical and retained the 39-build-hour cap: 13 hours per week through October 18, 2026. This plan supersedes the earlier single Isolation Forest experiment. The earlier plan is preserved in ARCHIVE_ISOLATION_FOREST_PLAN.md; SOURCE_CHAT.md remains historical background.

The initial boundary is reusable tasks on small numerical tables, spanning multiple application domains. Four task families and eight candidate adapters are the target. Breadth is bounded by shared input/output contracts and honest release gates. A catalog of many unverified tools would not satisfy this plan.

The current local folder is C:/Users/OpenClaw/Documents/ChatGPT/LLM Tool Calling Lab. The previous RLTTCMM path no longer exists. Keep artifacts here; do not create another project, chat, or online-only workspace.

## The central idea and the proof required

The question and the evidence should determine how the chatbot uses its models. A person describes what they want to learn; the system identifies the required answer type, checks available data, chooses eligible methods, runs the permitted analysis, and uses the results to decide what to say or do next.

Version 0.1 adapts the task/target interpretation, candidate choice, comparison and next step. The model architectures and small presets remain fixed. This is an external computational part of the system's reasoning, not a claim that the LLM's internal reasoning or weights are changed.

The four table tasks are the first test cases for this architecture. The product's reusable contribution is the common conversation, execution and extension path. Anomaly detection is one supported example rather than the identity of the whole project.

Demonstrate three things with artifacts already required by this plan: a question leads to an appropriate modeling approach; a meaningful follow-up affects a recorded action or conclusion; and replacing a model adapter changes a working example without rewriting the conversation layer. Performance comparisons establish where this implementation helps or fails.

The public working name is LLM Tool Calling Lab. MARKETING_STRATEGY.md describes how to present this concept and its evidence; it does not expand the build scope.

## What a person experiences

A person opens a local chat interface, chooses an included example or provides a CSV, and asks a question. The assistant identifies the kind of answer required, asks about missing facts in ordinary language, proposes suitable methods, runs the permitted comparisons, and answers using recorded results. The person can ask why a method was chosen, compare the alternative, or correct a goal.

Example questions:
- Assess numerical prediction using labeled examples and validation-row predictions.
- Assess binary category prediction using labeled examples and validation-row predictions.
- Rank a requested number of unusual cases relative to reference data.
- Group comparable cases into a requested number of exploratory groups.

The interface shows the answer first, then a compact evidence panel with the question as understood, required answer type, data roles, eligible/attempted methods, measured evidence, next action, assumptions and limitations. These are observable decisions and results, not a generated story about hidden internal reasoning. Algorithm names and run details are optional deeper information.

Ask people about facts and preferences they can supply, such as which outcome they mean or what a column represents. The person should not have to choose the algorithm. At least one demonstrated follow-up must cause an observable change: compare the remaining candidate, correct the target before fitting, retrieve evidence needed for a decision, or identify why an unsupported claim cannot be made. Merely rephrasing the previous answer does not demonstrate adaptation.

Use existing development examples for these demonstrations. A changed task begins a new bounded episode; make that boundary visible instead of quietly increasing the two-fit allowance.

Questions requiring unsupported evidence receive a specific explanation and a useful next step. Asking what would happen if an input changed can show a changed model prediction; it must not be presented as a proven causal effect.

## Architecture in plain language

1. The chat interface is the front desk. It accepts a question and local data and carries the conversation.
2. A planner is the coordinator. It turns the request into an explicit task and a short list of eligible approaches.
3. A model catalog is the toolbox. Each entry says what its model accepts, what it returns, and where it is appropriate.
4. An execution engine is the workshop. Ordinary code checks the request, keeps data partitions separate, fits allowed models, and returns structured results.
5. An evidence record is the lab notebook. It stores the configuration, tool calls, measured outputs, failures, and their identifiers.
6. The answer writer explains those results and proposes the next useful step. It uses the same configured LLM as the planner.

Conversation flows from question to plan to computation to evidence to answer, with a bounded loop for clarification or a second method. A second LLM, autonomous agent framework, and persistent personal memory are not required.

The planner proposes; execution code enforces supported operations and budgets. Model results are observations, not instructions. The LLM cannot read benchmark answer keys, silently change the scoring rules, or execute arbitrary generated code.

## Supported task and model contracts

| Task family | Initial candidate methods | Evidence available during conversation |
| --- | --- | --- |
| Numerical prediction | Ridge regression; small random forest regressor | Validation mean absolute error, predictions, comparison with a constant baseline |
| Binary category prediction | Logistic regression; small random forest classifier | Validation balanced accuracy, class counts, confusion matrix |
| Unusual cases | Isolation Forest; Local Outlier Factor in novelty mode | Rankings, reference suitability, score distributions and disagreement; no accuracy claim without labels |
| Exploratory grouping | K-means; Gaussian mixture model | Group summaries and common silhouette diagnostics; no claim that internal fit establishes meaningful real-world groups |

Each family has two genuine methods. Parameters use small documented presets; automatic architecture invention and broad hyperparameter search are outside version 0.1. A requested group count is shared by clustering candidates. If absent, the assistant asks or clearly labels a three-group exploratory default.

In this prototype, custom models means configurable models fitted to supplied data and a documented interface for adding a developer's own implementation. No claim is made that the system invents or trains new foundation models.

Numerical input features only; binary classification permits a target with exactly two categories. Initial operating limits: CSV up to 10 MB, 100 to 10,000 fitting/reference rows, 2 to 30 feature columns, stable row IDs, and explicit target/feature roles. Prediction/scoring batches may contain 1 to 10,000 rows; the 100-row minimum does not apply to those batches or to validation/test partitions. The developer may adjust limits on development examples before freezing them. Missing target values, insufficient class support, malformed inputs, and unsupported columns require explicit handling.

Independent rows are the supported prediction setting. Temporal forecasting, grouped observations requiring specialized splits, causal estimation, decision optimization, raw text, images, audio, and arbitrary web research are extension directions. The assistant must identify these limits rather than quietly treating those requests as supported predictions.

## Implementation choices for the later build

Proposed stack: Python, scikit-learn pipelines, typed request/result validation, a small Streamlit local chat UI, and local JSON run records. Confirm and pin compatible versions in the first build session.

A single provider adapter isolates the LLM backend. The first build session must prove one real tool-calling exchange with an available backend and record its exact model/settings. Prefer a local endpoint. Hosted execution is optional configuration and must clearly identify what information leaves the machine; raw data is not transmitted by default. Model summaries and tool outputs can still contain sensitive information.

No paid calls or training runs are authorized or performed by this planning task. The default paid-run budget is zero until the user sets a positive limit. A recorded replay is useful for inspecting the demo, but replay alone does not satisfy the live-chat acceptance criterion.

Proposed tool surface: inspect dataset, describe supported methods, run a candidate, compare compatible results, and retrieve a stored result. Model fitting, preprocessing, scoring and artifact creation happen behind validated interfaces.

Initial episode limits: at most two candidate fits, eight tool calls, six LLM responses including retries, and 180 seconds elapsed. Allow at most one repair attempt, inside the same limits. Count planning and answer-writing calls together. Freeze final token and spending caps after the backend smoke test and before held-out evaluation.

## Deliverables and definitions of done

| Deliverable | Definition of done |
| --- | --- |
| 1. Local conversational prototype | A clean installation runs a real LLM/tool conversation and a consequential follow-up, with visible question-to-method decisions and graceful failures. Each shipped family has an end-to-end example; at least one example records a follow-up changing an action or supported conclusion. |
| 2. Customizable model catalog | Eight target adapters use the same documented interface. A developer can change permitted models, presets and budgets through configuration. Every entry declares input requirements and output meaning. |
| 3. Examples and verification | Synthetic examples, data contracts and focused checks cover data separation, result validity, timeouts and missing inputs. Two development cases exist per target family. |
| 4. Reproducible comparison | Frozen held-out runs compare the structured planner with a generic LLM given the same tools, plus fixed deterministic family references. Raw records, denominators and failures are retained. |
| 5. Reuse documentation | Quickstart, architecture, supported-task table, model-extension guide, one exercised adapter replacement, example configuration and troubleshooting instructions are included. |
| 6. Open-source release package | Original code prepared under a proposed MIT license, third-party notices, dependency lock, secret-free example settings, reproducibility instructions, results and limitations. Publication is a later action after the build and checks. |

A template-only extension is insufficient: replace one existing adapter with a simple additional estimator using the documented interface, run the example, and record the result. The replacement need not join the main benchmark. Retain the changed catalog/configuration and before/after artifacts as the customization proof.

Reuse one verified development conversation and the adapter example for the later demo and case study. Label an illustrative demo separately from aggregate held-out findings. Capturing presentation assets belongs to the existing portfolio/writing allocation, not an additional application or unbudgeted build feature.

No improvement over a baseline is required to deliver useful work. A negative result with reproducible code and clear evidence fulfills the investigative purpose. Missing functionality does not become complete merely because the cap has been reached.

## Release gates and scope control

- At hour 13, each retained family runs both methods and produces a correct structured result on development examples. Defer a family whose contract remains unreliable; do not keep adding special cases indefinitely.
- At hour 24, each retained family completes a live conversation and handles its characteristic limitation correctly.
- At hour 26, freeze supported families, prompts, model presets, budgets, fixtures and scoring. Held-out evaluation begins only after this gate.
- At hour 35, the evaluation report and installable release package should be ready. Keep hours 35 to 39 for defects and clean-start verification.
- At hour 39, stop adding features. Record incomplete deliverables explicitly and make a continue, hold, or close decision from evidence.

Cut interface polish and extra examples first. Keep the shared extension contract, real chat path, evidence record, and evaluation. If necessary, defer an entire family before the freeze rather than weaken its checks. Statuses are implemented and evaluated, implemented but experimental, and planned or deferred. Report the actual support count.

If a family is deferred, revise benchmark denominators before held-out execution and preserve the decision. Do not drop difficult families after seeing their held-out results.

## Expected results and limits

Expected engineering outcome: a runnable reference that makes the architecture inspectable and allows people to replace the LLM, models, prompts, datasets and budgets without rebuilding the whole system.

Expected empirical outcome: evidence about which tasks the planner completes reliably, where simple workflows are sufficient, where model choices matter, and what extra latency/cost the conversation adds. Improved accuracy, broad transfer, lower expertise requirements and better human decisions remain hypotheses.

Generalizable means the same interfaces work across numerical-table datasets and four task families. Six held-out cases per family support exploratory findings, not universal claims. Synthetic findings do not establish production reliability.

A brief user walkthrough can identify confusing interactions. It is not evidence that the product broadly improves human reasoning; that would require a separate user study.

## Next build session

Only after moving into implementation: spend the first three-hour block fixing the data/task contracts, checking one real LLM tool call, and recording the environment and model choice. Then create the shared catalog and fixtures. Follow SCHEDULE.md; EVALUATION.md defines the comparison.

## Basis for the plan

These are design choices and estimates, not benchmark findings. Existing work establishes related capabilities:
- [HuggingGPT](https://arxiv.org/abs/2303.17580) selects and invokes specialist models.
- [Chameleon](https://arxiv.org/abs/2304.09842) composes tools for question answering.
- [AutoML-Agent](https://proceedings.mlr.press/v267/trirat25a.html) plans and verifies automated ML workflows.
- [scikit-learn evaluation documentation](https://scikit-learn.org/stable/model_selection.html) supplies the candidate library's evaluation mechanisms.
- [Streamlit chat components](https://docs.streamlit.io/develop/api-reference/chat) support the proposed small local interface.



