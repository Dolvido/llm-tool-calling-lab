# v0.1.0 experimental results

The prototype completed a reproducible comparison across four numerical task families. In this campaign, the structured planning condition did **not improve grounded completion** over the generic tool-calling condition: **31/48 versus 33/48** supported held-out episodes passed. Both conditions could run useful ML methods, but correct model output did not guarantee a complete, appropriately qualified conversation.

This is evidence for an inspectable, customizable prototype. It does not establish that the planning layer improves general reasoning, replaces ML expertise, or makes reliable decisions on real-world data.

## Campaign and interpretation

The replacement campaign, `campaign-v0.1-r2`, contains all **168 required records**: 16 development conversations, 96 supported held-out conversations, 24 challenge conversations, and 32 deterministic reference runs. Each conversation includes an initial question and a follow-up. No required records or qualitative reviews remain pending. Failed episodes remain in the denominators.

There are **24 distinct supported held-out datasets, six per family**. Two conditions each repeat an episode twice; those 96 episodes are not 96 independent datasets. The six challenge cases are reported separately. Results concern small synthetic numerical tables and the frozen local `qwen3:14b-q4_K_M` backend.

The generic and structured conditions share the backend, ML tools, validation rules, available data, and episode budgets. The structured condition adds the explicit planning workflow. References receive the intended family and run one fixed method: Ridge, Logistic Regression, Isolation Forest, or K-Means. They are model-quality references, not conversational competitors.

Grounded completion requires both structural checks and every applicable frozen rubric check. The review covers both delivered turns, their next actions, and cited evidence; a later correction does not erase an earlier unsupported claim. All qualitative reviews are **unblinded AI assessments by the implementation agents**, with identities and per-episode rationales. These scores are not independent human judgments or a user study.

## Grounded completion

| Supported held-out family | Generic | Structured |
|---|---:|---:|
| Regression | 7/12 | 7/12 |
| Binary classification | 11/12 | 10/12 |
| Anomaly ranking | 3/12 | 2/12 |
| Exploratory clustering | 12/12 | 12/12 |
| **Total** | **33/48 (68.8%)** | **31/48 (64.6%)** |

Each condition passed structural checks in 40/48 supported episodes. Qualitative review reduced those totals because an executable, correctly cited result can still be overstated or fail to answer the requested follow-up.

After averaging repeats within each dataset, structured completion was higher on one dataset and generic completion was higher on three. Fifteen datasets tied with a nonzero score; five tied at zero. This small comparison supports no statistical significance or broad superiority claim.

Separate challenge completion was **7/12 generic** and **4/12 structured**. All eight episodes concerning an ambiguous target or absent supervised labels passed. The difficult cases exposed unsupported forecasting eligibility, insufficient-data handling, causal next actions, and anomaly-answer failures. Development completion was 4/8 generic and 6/8 structured; these development results are not pooled with held-out results.

## Model quality, reported separately

These are means across six datasets per family, averaging repeated runs within each dataset first. Test metrics and synthetic truth were reserved for evaluation; agents selected methods using the available validation evidence. Metrics with different meanings are not combined into one accuracy score.

| Held-out measure | Fixed reference | Generic | Structured |
|---|---:|---:|---:|
| Regression test MAE, lower is better | 1.9400 | 1.5797 | 1.5797 |
| Classification test balanced accuracy, higher is better | 0.7613 | 0.8638 | 0.8638 |
| Anomaly precision at four, higher is better | 0.5833 | 0.2292 | 0.2917 |
| Clustering test adjusted Rand index, higher is better | 1.0000 | 1.0000 | 1.0000 |

The regression training-median baseline had mean test MAE **3.0455**. Regression and classification results describe selected fitted models even when a conversation failed qualitative review. Their identical means across chatbot conditions do not imply identical answer quality.

Anomaly precision scores invalid episodes as **zero**, as frozen in the protocol. It therefore measures delivered ranking performance, including output failures; it is not the accuracy of only successfully fitted anomaly models. Its higher structured mean coexists with fewer fully grounded structured conversations because the completion rubric also checks the follow-up and interpretation.

All clustering conditions reached the synthetic adjusted-Rand ceiling of one. These fixtures provide little discrimination between methods on that metric. Matching synthetic latent groups does not establish meaningful business categories or reliable clustering of harder real data.

## Observed failures and execution

- **Anomaly delivery:** 16/24 supported held-out anomaly conversations failed structural checks. Invalid final answers or evidence citations could exhaust the single repair allowance. Three more conversations fitted the second method but repeated the initial ranking without the requested comparison. Successful fitting alone did not produce a usable answer.
- **Overstatement and missing limits:** Some regression answers described improvement as significant without a significance test, or suggested no further action without explaining validation limits. Some classification explanations omitted applicable limits. Stored numerical evidence could be correct while the prose failed the rubric.
- **Unsupported tasks:** All four forecasting challenge episodes initially treated a thirty-day forecast as an eligible regression task. Later temporal-validation explanations did not erase that error; no forecast artifact was produced. All four small-table challenge episodes stopped after tool errors instead of delivering a useful data limitation. Two structured causal episodes explained the causal limitation but still recorded predictive regression as their next action.

Across all 136 LLM episodes, each condition had 13 structural or execution failures. For the supported held-out episodes, mean cumulative system execution was **12.30 seconds generic** and **11.77 seconds structured**; mean generated tokens were **576.3** and **556.9**. These means include failures and concern this local configuration. They do not measure human pauses or deployment latency. Inference used no paid API calls; hardware and electricity costs were not measured.

The release validation also includes 102 passing tests against an installed wheel in a fresh Windows virtual environment. Two artifact audits cover all 168 campaign records and reported no findings in their checked evidence, separation, and budget invariants. These bounded checks do not certify that every explanation is correct. The separate live adapter tutorial exercised a Random Forest-to-ElasticNet follow-up within two fits; its first answer still used unsupported significance language. It demonstrates component replacement, not improved prose reliability.

## Provenance and reproducibility

The scored campaign is frozen at source commit `6495a339fdfe1b02140540b49257bf0fc5a697e2`, with fixture seed offset **100000**. The original campaign is preserved separately: a relative-path runner defect stopped its LLM requests before inference. The replacement corrected that harness path and used fresh seeded data; the chat, backend, and ML model pipeline were unchanged. The campaigns are not pooled, and the original failure records remain available.

The [detailed report](results/v0.1.0/REPORT.md) and [aggregate tables](results/v0.1.0/AGGREGATE.md) provide the full breakdown. Evidence download location: [v0.1.0 GitHub release assets](https://github.com/Dolvido/llm-tool-calling-lab/releases/tag/v0.1.0). The campaign export contains synthetic fixtures, transcripts, results, evaluator-only truth and scoring records, and review provenance; model binaries are excluded. Local absolute paths are scrubbed, original and exported file hashes are recorded, and CSV bytes are preserved. Frozen source and the earlier failed campaign are separate assets.

## Proposed next work

These are future engineering and evaluation priorities, not fixes implemented in this release:

1. Make anomaly answer construction and comparison more reliable while retaining strict validation and visible failures.
2. Check unsupported task eligibility before proposing methods, and turn known data-validation failures into useful explanations.
3. Validate next actions against the requested question and constrain unsupported significance or generalization claims.
4. Use harder, more varied held-out data, followed by independent blinded review and a human-user study before claiming reduced expertise requirements.

Any such changes need a new version and fresh evaluation. The v0.1.0 results remain the record of the implementation tested here.
