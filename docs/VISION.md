# Vision and direction

LLM Tool Calling Lab is a local reference implementation for conversational use of numerical ML tools. I want developers to inspect how a question becomes a supported task, how a model is fitted, and whether the returned evidence actually supports the answer. A developer should also be able to replace a specialist without rebuilding the conversation engine.

The first published version is **v0.1.0 (experimental)**. “First release” describes the milestone; it is not a v1.0 stability claim. [Release and evidence assets](https://github.com/Dolvido/llm-tool-calling-lab/releases/tag/v0.1.0) · [Frozen results](https://github.com/Dolvido/llm-tool-calling-lab/blob/v0.1.0/RESULTS.md) · [Protocol](https://github.com/Dolvido/llm-tool-calling-lab/blob/v0.1.0/EVALUATION.md).

## What exists in v0.1.0

- Local Ollama/Streamlit conversations over numerical CSV data, using the tested `qwen3:14b-q4_K_M` backend.
- Regression, binary classification, anomaly ranking and exploratory clustering, with two eligible methods per family.
- An executor that owns preprocessing, data splits, fitting, timeouts and recorded artifacts. Evaluator-only labels and test scores stay outside chat tools.
- Follow-ups that share an enforced budget: two fits, eight tool calls, six LLM responses, one repair and 180 seconds of system work, with a 4,096-token generation cap.
- An exercised catalog replacement: Ridge to ElasticNet, including a fresh installed-wheel follow-up that compares Random Forest with ElasticNet.
- A complete frozen campaign and published failures, transcripts, review rationales, source snapshots and checksums.

Supervised and clustering outputs cover validation rows; separate fresh-batch scoring is available for anomalies. No model-generated code runs. The structured condition uses workflow instructions for the same LLM; it is not a trained router.

## What the first pass changed

The same backend, tools, data access and budgets produced **33/48 grounded completions for generic tool calling and 31/48 for structured orchestration** on supported held-out conversations. Added planning instructions did not improve the measured completion rate in this campaign.

Only **5/24 anomaly conversations** passed every check. Correct fitting could coexist with invalid outputs, missed comparisons or unsupported interpretation. The separate challenge suite also exposed forecasting eligibility, small-table explanations and inappropriate causal next actions.

These are results from **24 distinct synthetic held-out datasets**, each repeated across conditions. All 168 required records are complete: 16 development, 96 supported held-out and 24 challenge LLM episodes, plus 32 fixed references. The qualitative reviews are unblinded AI assessments by implementation agents, not independent human review. The small comparison supports no significance or general superiority claim.

My engineering decision is to retain the specialist interface, validated executor, shared budgets and inspectable artifacts. The extra planning prompt has not earned a claim of improved reliability. For a developer trying this version, the generic condition is the simpler comparison point; neither condition is established as reliable on real-world data.

## Proposed next slice

This is a prioritized backlog, not an implementation announcement or a promise of a v1.0 release. Keep one bounded change active and evaluate it in a new version.

| Priority | Observed problem | Definition of done for a later version |
| --- | --- | --- |
| 1. Anomaly delivery | Invalid selections and follow-ups that skip the requested comparison | Both turns return valid, cited selections; comparisons preserve the chosen artifact and explain disagreements. Invalid episodes remain in every relevant denominator. |
| 2. Task eligibility and data limits | Unsupported forecasts proposed as regression; tool errors without a useful explanation | Known unsupported requests are identified before a fit is proposed. Invalid-data cases explain the limitation and an appropriate next step under the existing budget. |
| 3. Evidence and next actions | Unsupported significance language, omitted limitations and predictive actions after causal refusals | Each claim and next action is checked against the question and recorded evidence. Correct numbers alone cannot make a conversation pass. |
| 4. Evaluation strength | Easy clustering fixtures and unblinded implementation-agent review | Fresh, more varied held-out cases and independent blinded review are defined before scoring. Any claim about reducing user expertise requires a separate human study. |

The expected win from the first slice is a more usable anomaly conversation under the same limits. Whether that happens is an evaluation question. No target success rate is promised.

Before changing prompts or answer construction, prepare a small development set drawn from the observed failure types and freeze its acceptance checks. The first ten-minute action is to inspect one invalid anomaly answer and one missed-comparison episode, then write the expected outputs.

## Evidence that stays fixed

The scored core is commit `6495a339fdfe1b02140540b49257bf0fc5a697e2`. The initial campaign failed before inference because of a relative-path harness defect. It is preserved separately; the replacement uses fresh seed offset `100000` with unchanged chat/model behavior. Their results are never pooled.

The released Windows installation passed 102 automated checks, and two artifact audits covered all 168 records without findings in their checked invariants. Those audits did not independently recompute every model metric. Successful installation, model execution and artifact checks do not establish that every explanation is correct.

Future changes require a new version, a new freeze and fresh evaluation. The v0.1.0 tag remains the record of this first pass. Forecasting, causal estimation, general reasoning, production reliability and reduced human expertise are not established by these results.

This direction does not expand the project into another general agent framework. Additional backends, hosted demos, installers and broader task families remain deferred.
