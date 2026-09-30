# Derived campaign summary

Campaign: `campaign-v0.1-r2`. Model: `qwen3:14b-q4_K_M`. Seed offset: `100000`.

**Execution complete:** 168/168 required records completed; 0 running; 0 missing. Completed structural successes awaiting qualitative review: 0.

This is a read-only derived snapshot, not a new inference run. Running campaigns can advance after records are read. JSON includes manifest and record hashes for the snapshot. Optional challenge-development episodes do not replace required episodes.

## Completion

Primary failure is zero; structurally successful but unreviewed is pending, never a pass. References have no conversational primary score. Counts use the frozen episode assignments.

| Split | Family | Condition | Expected | Completed | Structural passes | Primary passes | Primary failures | Pending review |
|---|---|---|---:|---:|---:|---:|---:|---:|
| challenge | challenge | generic | 12 | 12 | 9 | 7 | 5 | 0 |
| challenge | challenge | structured | 12 | 12 | 8 | 4 | 8 | 0 |
| development | anomaly | generic | 2 | 2 | 0 | 0 | 2 | 0 |
| development | anomaly | reference | 2 | 2 | 2 | N/A | N/A | 0 |
| development | anomaly | structured | 2 | 2 | 1 | 1 | 1 | 0 |
| development | classification | generic | 2 | 2 | 2 | 1 | 1 | 0 |
| development | classification | reference | 2 | 2 | 2 | N/A | N/A | 0 |
| development | classification | structured | 2 | 2 | 2 | 2 | 0 | 0 |
| development | clustering | generic | 2 | 2 | 2 | 2 | 0 | 0 |
| development | clustering | reference | 2 | 2 | 2 | N/A | N/A | 0 |
| development | clustering | structured | 2 | 2 | 2 | 2 | 0 | 0 |
| development | regression | generic | 2 | 2 | 2 | 1 | 1 | 0 |
| development | regression | reference | 2 | 2 | 2 | N/A | N/A | 0 |
| development | regression | structured | 2 | 2 | 2 | 1 | 1 | 0 |
| heldout | anomaly | generic | 12 | 12 | 4 | 3 | 9 | 0 |
| heldout | anomaly | reference | 6 | 6 | 6 | N/A | N/A | 0 |
| heldout | anomaly | structured | 12 | 12 | 4 | 2 | 10 | 0 |
| heldout | classification | generic | 12 | 12 | 12 | 11 | 1 | 0 |
| heldout | classification | reference | 6 | 6 | 6 | N/A | N/A | 0 |
| heldout | classification | structured | 12 | 12 | 12 | 10 | 2 | 0 |
| heldout | clustering | generic | 12 | 12 | 12 | 12 | 0 | 0 |
| heldout | clustering | reference | 6 | 6 | 6 | N/A | N/A | 0 |
| heldout | clustering | structured | 12 | 12 | 12 | 12 | 0 | 0 |
| heldout | regression | generic | 12 | 12 | 12 | 7 | 5 | 0 |
| heldout | regression | reference | 6 | 6 | 6 | N/A | N/A | 0 |
| heldout | regression | structured | 12 | 12 | 12 | 7 | 5 | 0 |

## Reviewed diagnostics

Only recorded rubric checks are aggregated. Appropriate task/target is one combined check. No unreviewed next-action taxonomy is inferred.
Pass counts below use only episodes where that check was explicitly reviewed. Machine failures and pending structural successes are shown separately; neither is silently counted as a reviewed success.

| Split | Family | Condition | Recorded check | Pass / reviewed | Machine failures | Pending structural successes |
|---|---|---|---|---:|---:|---:|
| challenge | challenge | generic | specific_limitation_or_clarification | 7 / 9 | 3 | 0 |
| challenge | challenge | generic | appropriate_next_step | 7 / 9 | 3 | 0 |
| challenge | challenge | generic | faithful_claims | 7 / 9 | 3 | 0 |
| challenge | challenge | structured | specific_limitation_or_clarification | 6 / 8 | 4 | 0 |
| challenge | challenge | structured | appropriate_next_step | 4 / 8 | 4 | 0 |
| challenge | challenge | structured | faithful_claims | 6 / 8 | 4 | 0 |
| development | anomaly | generic | appropriate_task_target | 0 / 0 | 2 | 0 |
| development | anomaly | generic | valid_required_output | 0 / 0 | 2 | 0 |
| development | anomaly | generic | faithful_claims | 0 / 0 | 2 | 0 |
| development | anomaly | generic | relevant_followup | 0 / 0 | 2 | 0 |
| development | anomaly | generic | appropriate_limits | 0 / 0 | 2 | 0 |
| development | anomaly | structured | appropriate_task_target | 1 / 1 | 1 | 0 |
| development | anomaly | structured | valid_required_output | 1 / 1 | 1 | 0 |
| development | anomaly | structured | faithful_claims | 1 / 1 | 1 | 0 |
| development | anomaly | structured | relevant_followup | 1 / 1 | 1 | 0 |
| development | anomaly | structured | appropriate_limits | 1 / 1 | 1 | 0 |
| development | classification | generic | appropriate_task_target | 2 / 2 | 0 | 0 |
| development | classification | generic | valid_required_output | 2 / 2 | 0 | 0 |
| development | classification | generic | faithful_claims | 2 / 2 | 0 | 0 |
| development | classification | generic | relevant_followup | 2 / 2 | 0 | 0 |
| development | classification | generic | appropriate_limits | 1 / 2 | 0 | 0 |
| development | classification | structured | appropriate_task_target | 2 / 2 | 0 | 0 |
| development | classification | structured | valid_required_output | 2 / 2 | 0 | 0 |
| development | classification | structured | faithful_claims | 2 / 2 | 0 | 0 |
| development | classification | structured | relevant_followup | 2 / 2 | 0 | 0 |
| development | classification | structured | appropriate_limits | 2 / 2 | 0 | 0 |
| development | clustering | generic | appropriate_task_target | 2 / 2 | 0 | 0 |
| development | clustering | generic | valid_required_output | 2 / 2 | 0 | 0 |
| development | clustering | generic | faithful_claims | 2 / 2 | 0 | 0 |
| development | clustering | generic | relevant_followup | 2 / 2 | 0 | 0 |
| development | clustering | generic | appropriate_limits | 2 / 2 | 0 | 0 |
| development | clustering | structured | appropriate_task_target | 2 / 2 | 0 | 0 |
| development | clustering | structured | valid_required_output | 2 / 2 | 0 | 0 |
| development | clustering | structured | faithful_claims | 2 / 2 | 0 | 0 |
| development | clustering | structured | relevant_followup | 2 / 2 | 0 | 0 |
| development | clustering | structured | appropriate_limits | 2 / 2 | 0 | 0 |
| development | regression | generic | appropriate_task_target | 2 / 2 | 0 | 0 |
| development | regression | generic | valid_required_output | 2 / 2 | 0 | 0 |
| development | regression | generic | faithful_claims | 1 / 2 | 0 | 0 |
| development | regression | generic | relevant_followup | 2 / 2 | 0 | 0 |
| development | regression | generic | appropriate_limits | 2 / 2 | 0 | 0 |
| development | regression | structured | appropriate_task_target | 2 / 2 | 0 | 0 |
| development | regression | structured | valid_required_output | 2 / 2 | 0 | 0 |
| development | regression | structured | faithful_claims | 1 / 2 | 0 | 0 |
| development | regression | structured | relevant_followup | 2 / 2 | 0 | 0 |
| development | regression | structured | appropriate_limits | 2 / 2 | 0 | 0 |
| heldout | anomaly | generic | appropriate_task_target | 4 / 4 | 8 | 0 |
| heldout | anomaly | generic | valid_required_output | 4 / 4 | 8 | 0 |
| heldout | anomaly | generic | faithful_claims | 4 / 4 | 8 | 0 |
| heldout | anomaly | generic | relevant_followup | 3 / 4 | 8 | 0 |
| heldout | anomaly | generic | appropriate_limits | 4 / 4 | 8 | 0 |
| heldout | anomaly | structured | appropriate_task_target | 4 / 4 | 8 | 0 |
| heldout | anomaly | structured | valid_required_output | 4 / 4 | 8 | 0 |
| heldout | anomaly | structured | faithful_claims | 4 / 4 | 8 | 0 |
| heldout | anomaly | structured | relevant_followup | 2 / 4 | 8 | 0 |
| heldout | anomaly | structured | appropriate_limits | 4 / 4 | 8 | 0 |
| heldout | classification | generic | appropriate_task_target | 12 / 12 | 0 | 0 |
| heldout | classification | generic | valid_required_output | 12 / 12 | 0 | 0 |
| heldout | classification | generic | faithful_claims | 12 / 12 | 0 | 0 |
| heldout | classification | generic | relevant_followup | 12 / 12 | 0 | 0 |
| heldout | classification | generic | appropriate_limits | 11 / 12 | 0 | 0 |
| heldout | classification | structured | appropriate_task_target | 12 / 12 | 0 | 0 |
| heldout | classification | structured | valid_required_output | 12 / 12 | 0 | 0 |
| heldout | classification | structured | faithful_claims | 12 / 12 | 0 | 0 |
| heldout | classification | structured | relevant_followup | 12 / 12 | 0 | 0 |
| heldout | classification | structured | appropriate_limits | 10 / 12 | 0 | 0 |
| heldout | clustering | generic | appropriate_task_target | 12 / 12 | 0 | 0 |
| heldout | clustering | generic | valid_required_output | 12 / 12 | 0 | 0 |
| heldout | clustering | generic | faithful_claims | 12 / 12 | 0 | 0 |
| heldout | clustering | generic | relevant_followup | 12 / 12 | 0 | 0 |
| heldout | clustering | generic | appropriate_limits | 12 / 12 | 0 | 0 |
| heldout | clustering | structured | appropriate_task_target | 12 / 12 | 0 | 0 |
| heldout | clustering | structured | valid_required_output | 12 / 12 | 0 | 0 |
| heldout | clustering | structured | faithful_claims | 12 / 12 | 0 | 0 |
| heldout | clustering | structured | relevant_followup | 12 / 12 | 0 | 0 |
| heldout | clustering | structured | appropriate_limits | 12 / 12 | 0 | 0 |
| heldout | regression | generic | appropriate_task_target | 12 / 12 | 0 | 0 |
| heldout | regression | generic | valid_required_output | 12 / 12 | 0 | 0 |
| heldout | regression | generic | faithful_claims | 7 / 12 | 0 | 0 |
| heldout | regression | generic | relevant_followup | 12 / 12 | 0 | 0 |
| heldout | regression | generic | appropriate_limits | 8 / 12 | 0 | 0 |
| heldout | regression | structured | appropriate_task_target | 12 / 12 | 0 | 0 |
| heldout | regression | structured | valid_required_output | 12 / 12 | 0 | 0 |
| heldout | regression | structured | faithful_claims | 9 / 12 | 0 | 0 |
| heldout | regression | structured | relevant_followup | 12 / 12 | 0 | 0 |
| heldout | regression | structured | appropriate_limits | 8 / 12 | 0 | 0 |

Detailed next-action assessment remains in `records/<episode_id>.json`: inspect both delivered responses, `next_action`, selected artifact IDs, and the identified review rationale. The session file under `artifacts/<episode_id>/sessions/` records actual tools. No comparison/clarification/refusal taxonomy was retrospectively guessed from wording.

## Usage across completed records

Means and ranges pool completed records across recorded splits by condition and are descriptive, not paired quality estimates. Missing counters are unavailable, not zero. Reference latency uses its recorded result duration; absent reference counters remain unavailable. Generated tokens include conservative reservations when usage was unknown; estimated-usage episodes are identified in JSON. Local hardware/electricity cost was not measured.

| Condition | Measure | Mean | Minimum | Maximum | Measured | Unavailable |
|---|---|---:|---:|---:|---:|---:|
| reference | Active seconds | 1.476 | 1.390 | 1.563 | 32 | 0 |
| reference | Candidate fits | unavailable | unavailable | unavailable | 0 | 32 |
| reference | Tool calls | unavailable | unavailable | unavailable | 0 | 32 |
| reference | LLM responses | unavailable | unavailable | unavailable | 0 | 32 |
| reference | Accounted generated tokens | unavailable | unavailable | unavailable | 0 | 32 |
| reference | Reported prompt/input tokens | unavailable | unavailable | unavailable | 0 | 32 |
| reference | Repairs | unavailable | unavailable | unavailable | 0 | 32 |
| generic | Active seconds | 11.269 | 3.016 | 19.203 | 68 | 0 |
| generic | Candidate fits | 1.397 | 0.000 | 2.000 | 68 | 0 |
| generic | Tool calls | 1.824 | 0.000 | 3.000 | 68 | 0 |
| generic | LLM responses | 4.029 | 2.000 | 6.000 | 68 | 0 |
| generic | Accounted generated tokens | 534.647 | 190.000 | 931.000 | 68 | 0 |
| generic | Reported prompt/input tokens | 14062.824 | 4960.000 | 26425.000 | 68 | 0 |
| generic | Repairs | 0.309 | 0.000 | 1.000 | 68 | 0 |
| structured | Active seconds | 10.922 | 3.093 | 19.547 | 68 | 0 |
| structured | Candidate fits | 1.368 | 0.000 | 2.000 | 68 | 0 |
| structured | Tool calls | 1.765 | 0.000 | 3.000 | 68 | 0 |
| structured | LLM responses | 3.956 | 2.000 | 6.000 | 68 | 0 |
| structured | Accounted generated tokens | 526.235 | 196.000 | 930.000 | 68 | 0 |
| structured | Reported prompt/input tokens | 14094.132 | 5160.000 | 26985.000 | 68 | 0 |
| structured | Repairs | 0.324 | 0.000 | 1.000 | 68 | 0 |

## Recorded structural or execution failures

Reasons below are the actual recorded final stop/error or structural-check messages. Recovered intermediate faults are not inferred as episode failures. Qualitative failures are represented in the review diagnostics above.

| Split | Family | Condition | Recorded reason | Episodes |
|---|---|---|---|---:|
| challenge | challenge | generic | invalid final answer or evidence citation | 1 |
| challenge | challenge | generic | tool execution failed; repair allowance exhausted | 2 |
| challenge | challenge | structured | invalid final answer or evidence citation | 2 |
| challenge | challenge | structured | tool execution failed; repair allowance exhausted | 2 |
| development | anomaly | generic | invalid final answer or evidence citation | 2 |
| development | anomaly | structured | invalid final answer or evidence citation | 1 |
| heldout | anomaly | generic | invalid final answer or evidence citation | 8 |
| heldout | anomaly | structured | invalid final answer or evidence citation | 8 |

## Held-out primary completion by case

The case mean is available only when every frozen repeat is complete and scored. A failed repeat contributes zero. Pending or missing repeats prevent a mean; they are not dropped. Repeats do not create independent datasets.

| Case | Family | Condition | Completed / expected repeats | Pending reviews | Primary mean |
|---|---|---|---:|---:|---:|
| c100 | regression | generic | 2 / 2 | 0 | 0.000 |
| c100 | regression | structured | 2 / 2 | 0 | 0.000 |
| c101 | regression | generic | 2 / 2 | 0 | 1.000 |
| c101 | regression | structured | 2 / 2 | 0 | 1.000 |
| c102 | regression | generic | 2 / 2 | 0 | 0.500 |
| c102 | regression | structured | 2 / 2 | 0 | 0.500 |
| c103 | regression | generic | 2 / 2 | 0 | 1.000 |
| c103 | regression | structured | 2 / 2 | 0 | 1.000 |
| c104 | regression | generic | 2 / 2 | 0 | 0.000 |
| c104 | regression | structured | 2 / 2 | 0 | 0.500 |
| c105 | regression | generic | 2 / 2 | 0 | 1.000 |
| c105 | regression | structured | 2 / 2 | 0 | 0.500 |
| c110 | classification | generic | 2 / 2 | 0 | 0.500 |
| c110 | classification | structured | 2 / 2 | 0 | 0.500 |
| c111 | classification | generic | 2 / 2 | 0 | 1.000 |
| c111 | classification | structured | 2 / 2 | 0 | 1.000 |
| c112 | classification | generic | 2 / 2 | 0 | 1.000 |
| c112 | classification | structured | 2 / 2 | 0 | 1.000 |
| c113 | classification | generic | 2 / 2 | 0 | 1.000 |
| c113 | classification | structured | 2 / 2 | 0 | 1.000 |
| c114 | classification | generic | 2 / 2 | 0 | 1.000 |
| c114 | classification | structured | 2 / 2 | 0 | 0.500 |
| c115 | classification | generic | 2 / 2 | 0 | 1.000 |
| c115 | classification | structured | 2 / 2 | 0 | 1.000 |
| c120 | anomaly | generic | 2 / 2 | 0 | 1.000 |
| c120 | anomaly | structured | 2 / 2 | 0 | 1.000 |
| c121 | anomaly | generic | 2 / 2 | 0 | 0.000 |
| c121 | anomaly | structured | 2 / 2 | 0 | 0.000 |
| c122 | anomaly | generic | 2 / 2 | 0 | 0.000 |
| c122 | anomaly | structured | 2 / 2 | 0 | 0.000 |
| c123 | anomaly | generic | 2 / 2 | 0 | 0.000 |
| c123 | anomaly | structured | 2 / 2 | 0 | 0.000 |
| c124 | anomaly | generic | 2 / 2 | 0 | 0.500 |
| c124 | anomaly | structured | 2 / 2 | 0 | 0.000 |
| c125 | anomaly | generic | 2 / 2 | 0 | 0.000 |
| c125 | anomaly | structured | 2 / 2 | 0 | 0.000 |
| c130 | clustering | generic | 2 / 2 | 0 | 1.000 |
| c130 | clustering | structured | 2 / 2 | 0 | 1.000 |
| c131 | clustering | generic | 2 / 2 | 0 | 1.000 |
| c131 | clustering | structured | 2 / 2 | 0 | 1.000 |
| c132 | clustering | generic | 2 / 2 | 0 | 1.000 |
| c132 | clustering | structured | 2 / 2 | 0 | 1.000 |
| c133 | clustering | generic | 2 / 2 | 0 | 1.000 |
| c133 | clustering | structured | 2 / 2 | 0 | 1.000 |
| c134 | clustering | generic | 2 / 2 | 0 | 1.000 |
| c134 | clustering | structured | 2 / 2 | 0 | 1.000 |
| c135 | clustering | generic | 2 / 2 | 0 | 1.000 |
| c135 | clustering | structured | 2 / 2 | 0 | 1.000 |

## Review provenance and limits

Identified unblinded AI assessment; not independent human review or a human-user study. Inspect per-episode rationales.

- AI: Codex ML implementation agent; rubric 0.1.0.
- AI: Codex chat implementation agent; rubric 0.1.0.
- AI: Codex primary agent; rubric 0.1.0.
- AI: Codex root, unblinded AI review of both delivered turns; rubric 0.1.0.

No causal, human-outcome, production-reliability, or general-superiority claim follows from these synthetic cases. Different families' model-quality metrics are not pooled by this script; inspect the campaign's original REPORT.md and per-episode metrics for those results.
