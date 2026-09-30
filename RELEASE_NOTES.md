# LLM Tool Calling Lab v0.1.0 — experimental

A local, open-source chatbot that chooses and compares numerical ML tools, displays their recorded evidence, and supports replacement models through a documented catalog interface.

## Included

- Four numerical task families and eight specialists: regression, binary classification, unusual-row ranking and exploratory grouping.
- Local Streamlit chat and native Ollama tool calling with cumulative episode limits, validated arguments, persistent evidence and consequential follow-ups.
- Windows/Python 3.12 quickstart, dependency lock, MIT original source, third-party notices and reproducible-failure issue template.
- Executed Ridge-to-ElasticNet tutorial, including a real follow-up from a freshly installed wheel.
- Full frozen campaign: 16 development, 96 supported held-out and 24 challenge LLM episodes, plus 32 fixed references. Failures and AI review rationales are retained.

## What the evaluation found

Supported held-out completion was **33/48 for the generic chatbot and 31/48 for structured orchestration**. The added planning instructions did not improve this measured completion rate. The same model, tools, data access and budgets were used in both conditions. Only 24 supported held-out datasets are independent; repeats are not extra datasets.

Anomaly conversations were unreliable: 5/24 passed every check, with invalid row IDs and missed follow-up comparisons prominent. Prose sometimes overstates evidence or omits limitations. Challenge success was 7/12 for generic and 4/12 for structured; the chatbot can propose an unsupported forecast despite later recognizing the temporal-validation problem. See [results](https://github.com/Dolvido/llm-tool-calling-lab/blob/v0.1.0/RESULTS.md) for separate model-quality metrics, diagnostics and limitations.

Qualitative scores are unblinded AI assessments by implementation agents. They are not independent human review and do not establish better human decisions, general reasoning, production reliability or real-world anomaly detection.

## Verification and provenance

The fresh Windows installed-wheel environment passed **102 automated checks**. Live local inference used `qwen3:14b-q4_K_M` on an RTX 5080 with 16 GB VRAM and approximately 32 GB RAM. A separate stored-artifact audit covered all 168 records without finding evidence mismatches, budget bypasses, partition overlap or exposed evaluator labels. The tagged commit must also pass the Windows CI workflow before publication.

The evaluated core is source commit `6495a339fdfe1b02140540b49257bf0fc5a697e2`. Later additions are documentation, evidence export/reporting utilities and their checks; the eight frozen core source hashes are unchanged. The first campaign's startup failure is preserved separately with its original source snapshot. The replacement campaign used fresh data with seed offset `100000`; its results are not pooled with that failed launch.

## Assets and use

Start with the README quickstart. Download the wheel or source distribution for packaging convenience, the evaluation ZIP for frozen fixtures/results/transcripts/reviews, and development evidence for actual demonstrations and earlier failures. Separate archives preserve the startup-failed campaign and both evaluated source snapshots. Verify downloads with `SHA256SUMS.txt`.

Model weights and fitted model binaries are excluded. Obtain the Ollama model separately. Supervised and clustering outputs currently cover validation rows; separate fresh-batch scoring is available for anomalies. Hosted demos, installers, other operating-system guarantees, forecasting, causality and additional backends are deferred.
