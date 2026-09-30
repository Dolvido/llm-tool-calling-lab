# v0.1 release status

The experimental v0.1.0 candidate satisfies the technical release gates. Publication is gated on the final commit passing the GitHub Windows workflow. The [GitHub release](https://github.com/Dolvido/llm-tool-calling-lab/releases/tag/v0.1.0) is the publication record.

| Gate | Verified evidence |
|---|---|
| At least two retained families, both methods working | All four retained; eight specialists tested and live four-family comparison demonstrations preserved |
| Live chat and consequential follow-up | Real local Qwen3 conversations; comparisons fit another method and can change the selected artifact |
| Complete frozen evaluation, failures retained | 168/168 required records: 16 development, 96 supported held-out, 24 challenge LLM episodes, plus 32 references; zero pending qualitative reviews |
| Exercised customization | Ridge/ElasticNet fitted tutorial; fresh installed-wheel chat actually compares Random Forest with ElasticNet |
| Integrity checks | 102 automated checks pass in a fresh Windows virtual environment; separate artifact audits cover all 168 records with no findings |
| Reusable release package | Locked dependencies, source, documentation, reports, synthetic evidence, preserved failed launch, checksums and issue template |

The [verification receipt](results/v0.1.0/verification.json) records counts and scope. The artifact audits check stored values, references, budgets, partition separation and evaluator-only data exposure. They do not independently recompute every model metric. GitHub's Windows workflow verifies installation, tests and build against the tagged source; the local live demonstration supplies the Ollama/GPU check.

Measured limitations are material. Supported held-out completion was 33/48 for the generic chatbot and 31/48 for structured orchestration. Only 5/24 anomaly conversations met all checks. Unsupported significance language, missing limitations, missed comparisons and unsupported forecasting proposals remain documented. Even the successful installed-wheel customization demonstration contains an unsupported significance claim; its execution success is not a qualitative pass. Read [RESULTS.md](RESULTS.md).

Every qualitative assessment is labeled AI and unblinded; no independent human evaluation or user study was performed. All source, backend, package and fixture fingerprints matched the freeze at the final verification. No evaluated core source changed after the replacement campaign began.

The initial startup-failed campaign is preserved separately. Its 136 LLM records failed before inference because of a relative-path harness defect. The replacement fixes that harness path and uses fresh seed offset 100000, with unchanged chatbot/model behavior. Results from the two campaigns are never pooled.

The October 18, 2026 target and 39 active development-hour cap were retained. Automated execution and concurrent AI work are not reported as human development hours; no human timesheet is inferred. The portfolio website and optional video remain outside this source-release delivery.
