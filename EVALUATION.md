# LLM Tool Calling Lab Evaluation Plan

Prepared September 29, 2026. The protocol is implemented and the full campaign is complete: 168 required records, no missing episodes, and no pending qualitative reviews. The current campaign is `.lab/campaign-v0.1-r2`, frozen at source commit `6495a339fdfe1b02140540b49257bf0fc5a697e2`, using all four families, the installed Qwen3 digest, and fresh seed offset 100000. Current results and release readiness are recorded in RELEASE_STATUS.md.

The first campaign is retained separately: a relative-path runner defect prevented every LLM request before inference. Its records, manifest, and exact source archive are preserved. The replacement fixes that harness path and uses fresh seeded data; it does not tune the model or prompts using held-out outcomes. These campaigns must never be pooled.

Qualitative reviews are explicitly identified, unblinded AI assessments with per-episode rationale. No independent human review or human-user study is claimed. Review both turns; a later correct answer does not erase an earlier unsupported claim. Structural validity and qualitative completion are reported separately.

## Questions the evaluation answers

Does an explicit task/model planning layer produce more valid, evidence-grounded answers than a generic tool-calling LLM given the same models and budget? How do its chosen model results compare with simple fixed workflows?

The evaluation measures operation, appropriate task interpretation, faithful use of evidence, model quality and cost. It does not measure general intelligence or establish that people without ML training make better real-world decisions.

## Comparison conditions

1. Fixed deterministic reference: one frozen method for the known family, with programmatic output. Use Ridge, logistic regression, Isolation Forest and K-means respectively. It receives the intended family and shared data splits; it is a model-quality reference, not a conversational competitor.
2. Generic tool-calling LLM: the same LLM, catalog descriptions, data access and validated tools, without the explicit structured planning workflow.
3. Proposed system: that LLM plus the structured task interpretation, candidate plan and evidence-based continuation workflow.

Both LLM conditions must use the same model version, sampling settings, per-episode total token limit, fit/tool-call limits, time limit, available schema information and repair allowance. Count any planner call within the total allowance. Both can access every catalog entry and appropriate tool evidence.

The shared engine enforces data separation in both conditions. The experiment tests the additional planning workflow rather than comparing safe tooling with an unconstrained system. Differences in prompts and control flow must be published.

### Implemented controls and limits of the comparison

Inspect `COMMON_PROMPT`, `STRUCTURED_PROMPT` and `Session` in [chat.py](src/llm_tool_calling_lab/chat.py). Generic already includes task eligibility, evidence, limitations and follow-up guidance; structured appends planning instructions to that common prompt. Both use the same session engine and catalog. [backend.py](src/llm_tool_calling_lab/backend.py) supplies the shared sampling settings. The frozen manifest, also embedded in [summary.json](results/v0.1.0/summary.json), records model digest `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`, temperature 0 and sampling seed 0. Repeat indices do not change that seed. Repeats measure repeated execution under these settings, not independent random draws.

Budgets are equal caps, not equal realized usage or equal prompt lengths. The planning addition changes input length, and the runner executes reference, generic, then structured within a case rather than randomizing order. Local latency is therefore descriptive, not an isolated causal speed comparison. Fixed references receive the intended family and use one predetermined method; the LLMs may fit two methods. Reference model-quality differences include method-selection and fit-count differences, not just conversational intelligence.

The two campaign source snapshots differ only in `evaluation.py` path resolution and `fixtures.py` seed-offset support. Offset 100000 changes numerical data **and seeded initial-question variant assignment**; it does not change the question-variant pool or index-based follow-up assignment. The failed launch produced no LLM inference. It is not a before/after performance comparison. Fresh seeds reduce direct reuse, but use the same known generator families and do not establish externally hidden evaluation data.

The saved rubric is version 0.1.0 with five supported checks and three challenge checks, defined in [evaluation.py](src/llm_tool_calling_lab/evaluation.py). Review histories identify AI reviewers, timestamps, boolean checks and rationale. Structural failures score zero without needing a qualitative review: 26 of the 136 LLM records have no review because they failed structurally; zero structurally successful episodes await review. Scores can vary with unblinded reviewer interpretation. No independent adjudication, inter-rater reliability, or registered external protocol is established.

Source and fixture hashes can be checked without inference. Saved installation and backend receipts support the evaluated environment; this documentation review does not recheck an installed model or package environment. See [the evidence map](RESULTS.md#inspect-the-evidence) for archive paths and source snapshots.

## Cases and counts

Target four families, with two development datasets and six held-out datasets per family: 8 development plus 24 held-out datasets. Each held-out dataset has a predetermined question and one follow-up. Each of the two LLM conditions repeats the entire episode twice.

- Initial development smoke campaign: 8 datasets x 2 LLM conditions x 1 run = 16 episodes. Additional development iterations must be logged; 16 is not an upper bound on tuning.
- Main held-out campaign: 24 datasets x 2 conditions x 2 repeats = 96 supported episodes.
- Separate challenge suite: 6 cases x 2 conditions x 2 repeats = 24 episodes.
- Initial campaign total: 136 LLM episodes, plus 8 development and 24 held-out deterministic reference runs. Count any additional development runs separately.

The main completion denominator is 48 episodes per LLM condition, or 12 per family. There are only 24 distinct held-out datasets, six per family. Repeats do not create independent datasets.

Develop and freeze generator families before held-out execution. Vary relationships, noise, imbalance and geometry as appropriate, and vary question wording independently. Use neutral filenames/IDs; do not expose generator names, expected winners or labels that answer the task.

Regression cases include additive and nonlinear relationships. Binary classification includes linear/nonlinear boundaries and varied balance; every valid case has exactly two target categories. Anomaly cases include isolated and local/correlated deviations. Clustering cases include different overlap and covariance structures. Include difficulty rather than tuning to make the proposed system win.

The common examples use small synthetic numeric tables. Supervised and clustering cases may use about 300 rows and 6 features. Anomaly cases use 256 reference rows and 64 new rows with exactly four planted unusual cases for the primary task. Exact generator settings and seeds are frozen in the implementation manifest.

## Separation of development, selection and final scoring

There are two distinct holdouts: unseen benchmark datasets, and test rows within a dataset.

For supervised tasks, use fixed 60/20/20 training/validation/test partitions, with valid stratification for classification. Fit preprocessing on training rows. Expose validation metrics to the agents; keep test targets inside the evaluator. Candidate selection must use validation evidence only. Do not refit after selection in version 0.1; evaluate the same fitted artifact on test features.

For clustering, fit both candidates on the training features. Predict validation assignments and compute the same silhouette diagnostic on identically transformed validation features. A candidate with an undefined silhouette is ineligible for diagnostic ranking. Latent group labels are evaluator-only and never enter fitting or method selection. Evaluate final predicted test groups with adjusted Rand index. A requested group count is part of the user's goal and may be supplied to both conditions; it is not inferred from hidden labels.

For anomaly detection, train on reference rows and score fresh batch rows. Local Outlier Factor must use novelty mode and must not use its novelty scoring path on its training rows. Hide planted anomaly labels from every tool and prompt. Agents may compare assumptions, rankings and disagreement, but cannot claim validation accuracy without labels. Retain the selected method ID and its ranking before the evaluator opens the labels.

The [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) describes why test data must stay out of fitting and model choices. The implementation must verify this separation.

## Primary completion metric

Score each supported episode as pass or fail using a frozen rubric. A pass requires all of the following:
- Appropriate supported task/target interpretation.
- A real successful model artifact within the episode budget.
- Required predictions, IDs or assignments in a valid schema.
- Numerical statements that agree with the cited tool artifact at the displayed precision.
- A relevant answer and follow-up that describe the applicable limitations without unsupported causal or certainty claims.

Schema/artifact/number checks should be automated. Review the remaining claims against a short rubric, with method labels hidden where practical. Record reviewer and rubric version. A single developer review is a limited assessment, not an independent human study.

Malformed outputs, exhausted retries, timeouts, unjustified refusals and invented results score zero and remain in the denominator. Record failure type and stage.

## Diagnostics explaining the planning layer

Report task/target interpretation and evidence-supported next action separately as pass/fail diagnostics extracted from the existing rubric and run records. They do not create new benchmark episodes or replace the primary completion score.

For the action diagnostic, record what evidence was available, which candidate or next step was chosen, and whether that choice was justified within the declared task contract. A model may be eligible without being demonstrated superior. Anomaly rankings without labels cannot justify an accuracy claim; a clustering diagnostic cannot establish real-world meaning.

Classify the existing follow-up outcomes: relevant comparison, clarification, evidence retrieval, justified limitation, unnecessary action or unsupported conclusion. Keep the user question, task interpretation and supporting artifact IDs together so the report can show where a failure occurred.

These diagnostics assess observed system behavior. They do not measure human comprehension or reveal the LLM's hidden reasoning.

## Model quality metrics

| Family | Conversation/selection evidence | Final benchmark measure |
| --- | --- | --- |
| Regression | Validation mean absolute error; constant baseline | Test mean absolute error and comparison with reference |
| Classification | Validation balanced accuracy | Test balanced accuracy; confusion matrix |
| Anomaly detection | Eligibility, rankings and disagreement only | Precision at 4 against planted labels |
| Clustering | Common validation silhouette, group summaries | Test adjusted Rand index against synthetic latent labels |

Do not average these unlike metrics into one accuracy score. Publish every family's results.

Require four distinct valid row IDs for the primary anomaly answer. Invalid output scores primary completion zero and anomaly precision at 4 zero. For other families, missing model-quality metrics are unavailable, accompanied by failure counts; do not invent a regression error or drop the episode silently. In paired comparison, a failed episode loses to a valid episode; two failures are recorded separately.

Silhouette measures a particular geometric notion of grouping and does not establish business meaning. The [clustering documentation](https://scikit-learn.org/stable/modules/clustering.html) describes methods and their assumptions. [Local Outlier Factor novelty detection](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html) explains the unseen-row restriction.

## Follow-ups and challenge cases

Every main episode includes one frozen follow-up. Use a balanced assignment across the existing cases to require a candidate comparison or a justified evidence-based next step; at least one development demonstration must change a recorded action or supported conclusion. A request to explain a metric is assessed for evidence use, but paraphrasing alone is not counted as adaptive behavior. The two-fit budget covers the whole episode. A new task starts a new episode; do not silently add budget mid-comparison.

The clarification and unsupported-request challenge cases provide further evidence about how conversation changes the task or response. Preserve the same follow-up wording, available evidence and total allowance across the two LLM conditions. These refinements do not add cases to the planned 136 initial episodes.

The separate six-case suite covers:
1. An ambiguous prediction target: ask a concrete clarification.
2. Missing labels for a supervised request: state what is needed.
3. Insufficient rows or invalid class support: report the data issue.
4. Unsupported causal/intervention question: explain the missing evidence.
5. Unsupported time-series or nonnumeric-feature request: describe the limitation.
6. A no-anomaly case paired with a request for certainty that everything is normal: avoid treating low relative scores as proof of absence.

Each challenge case has a development analogue and a held-out wording/data variant. Success means identifying the actual limitation and an appropriate next step; a generic refusal alone is insufficient. Report challenge success separately out of 12 episodes per LLM condition.

Independent executable checks should inject an invalid tool argument and a model timeout, verify validation and budget behavior, and confirm artifacts are never fabricated. These checks are not extra benchmark datasets.

## Runtime, usage and freeze record

Record model and package versions, case ID, seed, prompts, task plan, candidate IDs/settings, data-partition fingerprints, tool outputs, artifact IDs, retries, errors, elapsed time and usage. Record cost when known; otherwise mark it unavailable. Local inference is not costless simply because no API charge is recorded.

The frozen cap is two candidate fits, eight tool calls, six LLM responses, one repair attempt inside those limits and 180 seconds cumulative system execution per episode, excluding human pauses. Follow-ups share the same allowance. The model context is 8192 tokens; generation is bounded to 1024 per response and 4096 per episode, including any reasoning tokens. Inference is local, with zero paid API calls and unmeasured hardware/electricity cost. No scope reduction was made for the replacement campaign.

At the freeze, save a manifest of the supported-family list, datasets/generators, prompts, schema, model presets, backend, budgets, metrics, rubrics and case assignments. Final labels cannot be used for prompt revision, model selection or score-driven reruns.

## Reporting and decisions

Report completion/failure rates, paired wins/ties/losses, per-family model quality, faithfulness findings, tool/fit counts, latency and available cost. Average repeats within a case for paired dataset-level comparisons. Show case-level results; do not claim significance from treating repeats as independent cases.

If the structured layer helps, describe where and at what overhead. If it ties or loses, explain why and retain the results. A useful open-source prototype does not require a positive accuracy finding.

If a family is deferred before the freeze, change the supported denominator to 6 x retained families x 2 conditions x 2 repeats and show the support change. Keep challenge results separate. Do not remove a family after seeing its held-out failures to improve the reported score.

No generalization claim beyond the tested numerical task families and synthetic settings is established by this campaign. A small walkthrough is usability feedback only; measuring improved human understanding is a later study.



