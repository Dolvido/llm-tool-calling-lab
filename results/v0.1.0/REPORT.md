# Evaluation results

Experimental synthetic-data evaluation. Repeats are not independent datasets.

Model: `qwen3:14b-q4_K_M`. Families frozen: regression, classification, anomaly, clustering.

Initial protocol: 168/168 episode records, including deterministic references; **0 required episodes missing**. Optional challenge-development runs are additional and do not replace required episodes.

Missing qualitative reviews remain unscored. AI review, if present, is identified and is not an independent human study.

| Split | Family | Condition | Expected | Recorded | Missing | Structural passes | Primary passes | Pending review |
|---|---|---|---:|---:|---:|---:|---:|---:|
| challenge | challenge | generic | 12 | 12 | 0 | 9 | 7 | 0 |
| challenge | challenge | structured | 12 | 12 | 0 | 8 | 4 | 0 |
| development | anomaly | generic | 2 | 2 | 0 | 0 | 0 | 0 |
| development | anomaly | reference | 2 | 2 | 0 | 2 | N/A | 0 |
| development | anomaly | structured | 2 | 2 | 0 | 1 | 1 | 0 |
| development | classification | generic | 2 | 2 | 0 | 2 | 1 | 0 |
| development | classification | reference | 2 | 2 | 0 | 2 | N/A | 0 |
| development | classification | structured | 2 | 2 | 0 | 2 | 2 | 0 |
| development | clustering | generic | 2 | 2 | 0 | 2 | 2 | 0 |
| development | clustering | reference | 2 | 2 | 0 | 2 | N/A | 0 |
| development | clustering | structured | 2 | 2 | 0 | 2 | 2 | 0 |
| development | regression | generic | 2 | 2 | 0 | 2 | 1 | 0 |
| development | regression | reference | 2 | 2 | 0 | 2 | N/A | 0 |
| development | regression | structured | 2 | 2 | 0 | 2 | 1 | 0 |
| heldout | anomaly | generic | 12 | 12 | 0 | 4 | 3 | 0 |
| heldout | anomaly | reference | 6 | 6 | 0 | 6 | N/A | 0 |
| heldout | anomaly | structured | 12 | 12 | 0 | 4 | 2 | 0 |
| heldout | classification | generic | 12 | 12 | 0 | 12 | 11 | 0 |
| heldout | classification | reference | 6 | 6 | 0 | 6 | N/A | 0 |
| heldout | classification | structured | 12 | 12 | 0 | 12 | 10 | 0 |
| heldout | clustering | generic | 12 | 12 | 0 | 12 | 12 | 0 |
| heldout | clustering | reference | 6 | 6 | 0 | 6 | N/A | 0 |
| heldout | clustering | structured | 12 | 12 | 0 | 12 | 12 | 0 |
| heldout | regression | generic | 12 | 12 | 0 | 12 | 7 | 0 |
| heldout | regression | reference | 6 | 6 | 0 | 6 | N/A | 0 |
| heldout | regression | structured | 12 | 12 | 0 | 12 | 7 | 0 |

## Paired completion by case

Repeated primary-completion scores are averaged inside each case before comparing conditions. A failed episode scores zero; unreviewed episodes remain pending. Cases missing either arm or repeat are incomplete. Both-zero cases are shown separately from ties. No significance claim is made.

| Split | Family | Structured wins | Generic wins | Ties | Both fail | Pending review | Incomplete |
|---|---|---:|---:|---:|---:|---:|---:|
| challenge | challenge | 0 | 2 | 2 | 2 | 0 | 0 |
| development | anomaly | 1 | 0 | 0 | 1 | 0 | 0 |
| development | classification | 1 | 0 | 1 | 0 | 0 | 0 |
| development | clustering | 0 | 0 | 2 | 0 | 0 | 0 |
| development | regression | 0 | 0 | 1 | 1 | 0 | 0 |
| heldout | anomaly | 0 | 1 | 1 | 4 | 0 | 0 |
| heldout | classification | 0 | 1 | 5 | 0 | 0 | 0 |
| heldout | clustering | 0 | 0 | 6 | 0 | 0 | 0 |
| heldout | regression | 1 | 1 | 3 | 1 | 0 | 0 |

## Model quality

Metrics are averaged within a case before averaging cases. Missing metrics are unavailable, not silently treated as successes. Inspect failures above and episode records.

| Split | Family | Condition | Metric | Case mean | Cases with metric |
|---|---|---|---|---:|---:|
| development | anomaly | generic | precision_at_4 | 0.0000 | 2 |
| development | anomaly | reference | precision_at_4 | 0.6250 | 2 |
| development | anomaly | structured | precision_at_4 | 0.5000 | 2 |
| development | classification | generic | test_balanced_accuracy | 0.8986 | 2 |
| development | classification | reference | test_balanced_accuracy | 0.6556 | 2 |
| development | classification | structured | test_balanced_accuracy | 0.8986 | 2 |
| development | clustering | generic | test_adjusted_rand_index | 1.0000 | 2 |
| development | clustering | reference | test_adjusted_rand_index | 1.0000 | 2 |
| development | clustering | structured | test_adjusted_rand_index | 1.0000 | 2 |
| development | regression | generic | baseline_test_mae | 3.0397 | 2 |
| development | regression | generic | test_mae | 1.7304 | 2 |
| development | regression | reference | baseline_test_mae | 3.0397 | 2 |
| development | regression | reference | test_mae | 1.7304 | 2 |
| development | regression | structured | baseline_test_mae | 3.0397 | 2 |
| development | regression | structured | test_mae | 1.7304 | 2 |
| heldout | anomaly | generic | precision_at_4 | 0.2292 | 6 |
| heldout | anomaly | reference | precision_at_4 | 0.5833 | 6 |
| heldout | anomaly | structured | precision_at_4 | 0.2917 | 6 |
| heldout | classification | generic | test_balanced_accuracy | 0.8638 | 6 |
| heldout | classification | reference | test_balanced_accuracy | 0.7613 | 6 |
| heldout | classification | structured | test_balanced_accuracy | 0.8638 | 6 |
| heldout | clustering | generic | test_adjusted_rand_index | 1.0000 | 6 |
| heldout | clustering | reference | test_adjusted_rand_index | 1.0000 | 6 |
| heldout | clustering | structured | test_adjusted_rand_index | 1.0000 | 6 |
| heldout | regression | generic | baseline_test_mae | 3.0455 | 6 |
| heldout | regression | generic | test_mae | 1.5797 | 6 |
| heldout | regression | reference | baseline_test_mae | 3.0455 | 6 |
| heldout | regression | reference | test_mae | 1.9400 | 6 |
| heldout | regression | structured | baseline_test_mae | 3.0455 | 6 |
| heldout | regression | structured | test_mae | 1.5797 | 6 |

## Usage and failures

Local inference incurred no paid API calls; electricity and hardware costs were not measured.

- generic: 68 recorded episodes; mean active seconds 11.27 (68 measured, 0 unavailable); 13 structural/execution failures.
- structured: 68 recorded episodes; mean active seconds 10.92 (68 measured, 0 unavailable); 13 structural/execution failures.

## Review provenance

- AI: Codex ML implementation agent
- AI: Codex chat implementation agent
- AI: Codex primary agent
- AI: Codex root, unblinded AI review of both delivered turns

## Limits

These results concern small synthetic numerical tables and this frozen backend. They do not establish production reliability, causal inference, general intelligence, or reduced human expertise. Failed episodes remain in records and denominators. No superiority claim is inferred from this report.
